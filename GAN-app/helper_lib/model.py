import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()

        self.fc_block = nn.Linear(z_dim, 128 * 7 * 7)

        self.conv_block = nn.Sequential(
            # Input: 128 x 7 x 7
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Shape becomes: 64 x 14 x 14

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,  # 1 channel for grayscale
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
            # Final output shape: 1 x 28 x 28
        )
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

    def forward(self, x):
        x = self.fc_block(x)
        # Reshape to 7x7x128 feature map
        x = x.view(-1, 128, 7, 7)
        # Pass through convolutional upsampling block
        x = self.conv_block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2),
            # Shape becomes: 64 x 14 x 14

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # Shape becomes: 128 x 7 x 7

            nn.Flatten(),

            # The input features are 128 * 7 * 7
            nn.Linear(
                in_features=128 * 7 * 7,
                out_features=1  # Single output (real/fake probability)
            )
        )

    def forward(self, x):
        x = self.model(x)
        return x
