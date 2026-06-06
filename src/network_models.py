import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, num_classes=3, input_size=42):
        super().__init__()

        self.layers = nn.Sequential(
            # layer 1
            nn.Linear(in_features=input_size, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # layer 2
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # last layer, output
            nn.Linear(in_features=32, out_features=num_classes)#joinha, point up, nada
        )

    def forward(self, x):
        output = self.layers(x)
        return output
    
    

class DeepTestModel(nn.Module):
    def __init__(self, num_classes=3, input_size=42):
        super().__init__()

        self.layers = nn.Sequential(
            # layer 1
            nn.Linear(in_features=input_size, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # layer 2
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # layer 3
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),

            # last layer, output
            nn.Linear(in_features=32, out_features=num_classes)#joinha, point up, nada
        )

    def forward(self, x):
        output = self.layers(x)
        return output
