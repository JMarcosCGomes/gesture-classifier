import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, num_classes, input_size):
        super().__init__()

        self.layers = nn.Linear(in_features=input_size, out_features=num_classes)

    def forward(self, x):
        output = self.layers(x)
        return output
    

class OneLayerModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden=16, dropout=0.2):
        super().__init__()

        self.layers = nn.Sequential(
            # layer 1
            nn.Linear(in_features=input_size, out_features=hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            # last layer, output
            nn.Linear(in_features=hidden, out_features=num_classes)
        )

    def forward(self, x):
        output = self.layers(x)
        return output
    

class TwoLayerModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden=(16, 8), dropout=0.2):
        super().__init__()
        h1, h2 = hidden
        self.layers = nn.Sequential(
            # layer 1
            nn.Linear(in_features=input_size, out_features=h1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # layer 2
            nn.Linear(in_features=h1, out_features=h2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # output
            nn.Linear(in_features=h2, out_features=num_classes)
        )

    def forward(self, x):
        output = self.layers(x)
        return output
