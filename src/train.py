import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader, random_split
from src.dataset import GestureDataset

from src.network_models import SimpleModel

def main():
    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent
    CSV_PATH = PROJECT_ROOT / 'data' / 'processed' / 'processed_landmarks.csv'
    MODELS_DIR = PROJECT_ROOT / 'models'
    LOGS_DIR = PROJECT_ROOT / 'logs'

    # --- Run ID --- run id using datetime, to keep unique model name
    run_id = datetime.now().strftime("run%Y%m%d_%H%M%S")
    run_log_dir = LOGS_DIR / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    dataset = GestureDataset(CSV_PATH)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    
    # --- Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = SimpleModel().to(device)

    # --- Training ---
    criteria = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training initialized..")
    epochs = 500
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODELS_DIR / f'{run_id}_best.pth')


    torch.save(model.state_dict(), MODELS_DIR / f'{run_id}_last.pth')
    print("Training finished..")


if __name__ == "__main__":
    main()