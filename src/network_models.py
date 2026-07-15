import torch
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


class RBFLayer(nn.Module):
    def __init__(self, input_size, num_centers, centers=None):
        super().__init__()
        self.centers = nn.Parameter(torch.empty(num_centers, input_size))
        self.log_betas = nn.Parameter(torch.empty(num_centers))
        self._initialize_weights(centers)

    def _initialize_weights(self, centers=None):
        if centers is None:
            nn.init.uniform_(self.centers, -1.0, 1.0)
        else:
            with torch.no_grad():
                self.centers.copy_(centers)
        nn.init.constant_(self.log_betas, 0.0)

    def forward(self, x):
        beta = torch.exp(self.log_betas).unsqueeze(0) # (1, num_centers)
        x_expanded = x.unsqueeze(1) # (batch_size, 1, input_size)
        centers_expanded = self.centers.unsqueeze(0) # (1, num_centers, input_size)
        # ||x - c_j||^2
        distances = torch.sum((x_expanded - centers_expanded) ** 2, dim=2) # (batch_size, num_centers)
        output = torch.exp(-beta * distances)
        return output


class RBFModel(nn.Module):
    def __init__(self, num_classes, input_size, centers=None):
        super().__init__()
        num_centers = 3 * num_classes
        self.rbf = RBFLayer(input_size, num_centers, centers)
        self.classifier = nn.Linear(num_centers, num_classes)
        
    def forward(self, x):
        features = self.rbf(x)
        output = self.classifier(features)
        return output