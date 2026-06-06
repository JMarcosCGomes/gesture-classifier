import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


class GestureDataset(Dataset):
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)

        unique_labels = sorted(df['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}

        self.X = torch.tensor(df.drop(columns='label').values, dtype=torch.float32)
        self.y = torch.tensor([self.label_to_idx[label] for label in df['label']], dtype=torch.long)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]