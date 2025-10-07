import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Shape becomes: 16 x 32 x 32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # Shape becomes: 32 x 16 x 16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # The input features for the linear layer are 32  * 16 * 16
            nn.Linear(in_features=32 * 16 * 16, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=num_classes)
        )

    def forward(self, x):
        """The forward pass of the model."""
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(model_name):
    model = CNN(num_classes=10)
    return model