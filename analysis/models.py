import torch
import torch.nn as nn

class EEGCNN(nn.Module):
    def __init__(self, filter_size: int = 3, num_classes: int = 108):
        """
        Args:
            filter_size: 3 (C3), 6 (C6), or 8 (C8)
            num_classes: 4 (output classes)
        """
        super().__init__()
        
        # Convolutional layer with square filter
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=filter_size, padding=filter_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=filter_size, padding=filter_size // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # Downsample to reduce spatial dimensions
        )
        
        # Fully Connected Layers for regression
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),  # Flattened size after pooling = 32 * 9 * 9
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)  # Final output size matches the labels
        )

    def forward(self, x):
        # Input shape: (batch, 1, 19, 19)
        x = self.conv(x)            # Still (batch, 1, 19, 19)
        x = self.classifier(x)              # Output: (batch, 4)
        return x
