from torch import nn


class Model(nn.Module):
    def __init__(self, num_features=784, num_classes=10) -> None:
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        features = self.features(x)
        probs = self.classifier(features)
        
        return probs