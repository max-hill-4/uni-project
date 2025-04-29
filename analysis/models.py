import torch
import torch.nn as nn
class EEGCNN(nn.Module):
    def __init__(self, in_channels: int = 1, filter_size: int = 3, num_classes: int = 108):
        """
        Args:
            in_channels: Number of input channels (e.g., 1 for single-channel, 5 for multi-channel).
            filter_size: 3 (C3), 6 (C6), or 8 (C8).
            num_classes: Number of output classes.
        """
        super().__init__()
        
        # Convolutional layers (now supports multi-channel input)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=filter_size, padding=filter_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=filter_size, padding=filter_size // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # Reduces spatial dims from 19x19 to 9x9
        )
        
        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch, in_channels, 19, 19)
        x = self.conv(x)            # Output: (batch, 32, 9, 9)
        x = self.classifier(x)      # Output: (batch, num_classes)
        return x