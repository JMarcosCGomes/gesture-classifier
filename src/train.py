import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from torch.utils.data import DataLoader
from src.dataset import GestureDataset

from src.network_models import SimpleModel

def main():
    SRC_PATH = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_PATH.parent
    CSV_PATH = PROJECT_ROOT / 'data' / 'processed' / 'processed_landmarks.csv'
    MODELS_DIR = PROJECT_ROOT / 'models'
    dataset = GestureDataset(CSV_PATH)
    loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = SimpleModel().to(device)

    criteria = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training initialized..")
    epochs = 500
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad() #clears the gradients from the previous batch
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            loss = criteria(predictions, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (predictions.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        if(epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODELS_DIR / 'best_model.pth')

    torch.save(model.state_dict(), MODELS_DIR / 'last_model.pth')
    print("Training finished..")


if __name__ == "__main__":
    main()