import argparse
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from src.dataset import GestureDataset
from src.network_models import LinearModel, OneLayerModel, TwoLayerModel, RBFModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["landmarks", "landmarksreduced", "angles", "anglesreduced", "distances", "curlnabductionmcpntipttip", "curlnabductionmcp", "curlnpinch", "jointnpinch"], default="landmarks")
    parser.add_argument("--model", choices=["linear", "onelayer", "twolayer", "rbf"], default="twolayer")
    parser.add_argument("--train-frac", type=float, default=1.0, help="quantos '%' do dataset vai ser usado no treino")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout do networkmodel")
    args = parser.parse_args()

    DATASETS = {
    "landmarks": "processed_landmarks.csv",
    "landmarksreduced": "landmarks_reduced.csv",
    "angles": "featured_angles.csv",
    "anglesreduced": "featured_angles_reduced.csv",
    "distances": "featured_distances.csv",
    "curlnabductionmcpntipttip": "combined_curl_abductionmcp_tipttip.csv",
    "curlnabductionmcp": "combined_curl_abductionmcp.csv",
    "curlnpinch": "combined_curl_pinch.csv",
    "jointnpinch": "combined_joint_pinch.csv",
    }
    
    #acho que da ate pra tirar isso aqui
    MODELS = {
    "linear": LinearModel,
    "onelayer": OneLayerModel,
    "twolayer": TwoLayerModel,
    "rbf": RBFModel,
    }

    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent
    CSV_PATH = PROJECT_ROOT / 'data' / 'processed' / DATASETS[args.dataset]
    MODELS_DIR = PROJECT_ROOT / 'models'
    LOGS_DIR = PROJECT_ROOT / 'logs'

    # --- Dataset ---
    dataset = GestureDataset(CSV_PATH)
    dataset_name = CSV_PATH.stem
    idxs = list(range(len(dataset)))
    labels = [dataset[i][1].item() for i in idxs] # used in stratify
    input_size = dataset.X.shape[1]  # used in model
    num_classes = len(dataset.label_to_idx) # used in model

    trainval_idx, test_idx = train_test_split(idxs, test_size=0.15, random_state=42, stratify=labels)
    trainval_labels = [labels[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.176, random_state=42, stratify=trainval_labels) # 0.176*85~=0.15
    
    #pos split faz a possivel diminuição nos dados de treino
    if args.train_frac < 1.0:
        train_labels = [labels[i] for i in train_idx]
        train_idx, _ = train_test_split(train_idx, train_size=args.train_frac, random_state=42, stratify=train_labels)
        print(f"Using {len(train_idx)} train samples ({args.train_frac*100:.0f}%)")


    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # --- Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    match args.model:
        case "linear":
            model = LinearModel(
                num_classes=num_classes,
                input_size=input_size,
            ).to(device)
        case "onelayer":
            model = OneLayerModel(
                num_classes=num_classes,
                input_size=input_size,
                dropout=args.dropout,
            ).to(device)
        case "twolayer":
            model = TwoLayerModel(
                num_classes=num_classes,
                input_size=input_size,
                dropout=args.dropout,
            ).to(device)
        case "rbf":
            #TODO: decidir se vai só usar kmeans ou vai criar arg pra kmeans ou random. deixei no modelo uma base tranquila.
            num_centers = 3 * num_classes #isso aqui ta errado mas é o que tem pra hoje. É a mesma conta feita no rbfmodel já
            kmeans = KMeans(n_clusters=num_centers, random_state=42, n_init="auto")
            X_train = dataset.X[train_idx]
            kmeans.fit(X_train.numpy())
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            model = RBFModel(
                num_classes=num_classes,
                input_size=input_size,
                centers=centers,
            ).to(device)
    model_name = model.__class__.__name__

    # --- Run ID --- run id using datetime, to keep unique model name
    now = datetime.now().strftime("run%Y%m%d_%H%M%S")
    run_id = f'{now}_{dataset_name}_{model_name}_{args.train_frac}'
    run_log_dir = LOGS_DIR / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # --- Metrics CSV ---
    metrics_path = run_log_dir / 'metrics.csv'
    metrics_file = open(metrics_path, 'w', newline='')
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
    metrics_writer.writeheader()

    # --- Training ---
    criteria = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training initialized..")
    epochs = 500
    PATIENCE = 30
    patience_counter = 0
    DELTA = 1e-4
    best_val_loss = float('inf')
    for epoch in range(epochs):

        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad() #clears the gradients from the previous batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            loss = criteria(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (predictions.argmax(dim=1) == y_batch).sum().item()
            train_total += y_batch.size(0)
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                loss = criteria(predictions, y_batch)

                val_loss += loss.item()
                val_correct += (predictions.argmax(dim=1) == y_batch).sum().item()
                val_total += y_batch.size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        if(epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"Train - Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.4f}")
            print(f"Val - Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f}")

        if avg_val_loss < (best_val_loss - DELTA):
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f'{run_id}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at {epoch + 1}, patience = {PATIENCE} epochs")
                break

        # Log metrics
        metrics_writer.writerow({
            'epoch': epoch + 1,
            'train_loss': round(avg_train_loss, 6),
            'val_loss': round(avg_val_loss, 6),
            'train_accuracy': round(train_accuracy, 6),
            'val_accuracy': round(val_accuracy, 6)
        })


    torch.save(model.state_dict(), MODELS_DIR / f'{run_id}_last.pth')
    metrics_file.close()
    print("Training finished..")

    print("Testing initialized...")
    all_preds = []
    all_labels = []
    model.load_state_dict(torch.load(MODELS_DIR / f'{run_id}_best.pth'))
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            loss = criteria(predictions, y_batch)

            all_preds.extend(predictions.argmax(dim=1).cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

            test_loss += loss.item()
            test_correct += (predictions.argmax(dim=1) == y_batch).sum().item()
            test_total += y_batch.size(0)
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total

    test_results = {
        'run_id': run_id,
        'model_name': model_name,
        'input_size': input_size,
        'num_classes': num_classes,
        'csv_path': str(CSV_PATH),
        'train_frac': args.train_frac,
        'dropout': args.dropout,
        'test_loss': round(avg_test_loss, 6),
        'test_accuracy': round(test_accuracy, 6),
        'best_val_loss': round(best_val_loss, 6),
        'stopped_early': patience_counter >= PATIENCE,
        'predictions': all_preds,
        'labels': all_labels,
        'class_names': dataset.idx_to_label,
    }

    with open(run_log_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    print("Testing finished...")

if __name__ == "__main__":
    main()