import torch
import torch.nn as nn

class EEGCNN(nn.Module):
    def __init__(self, filter_size: int = 3, num_classes: int = 4):
        """
        Args:
            filter_size: 3 (C3), 6 (C6), or 8 (C8)
            num_classes: 4 (output classes)
        """
        super().__init__()
        
        # Convolutional layer with square filter
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=filter_size, padding=filter_size//2),
            nn.ReLU(inplace=True)  # ReLU activation
        )
        
        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(19 * 19, num_classes)  # Output = num_classes
        )
    def forward(self, x):
        # Input shape: (batch, 1, 19, 19)
        x = self.conv(x)            # Still (batch, 1, 19, 19)
        x = x.view(x.size(0), -1)   # Flatten to (batch, 361)
        x = self.classifier(x)              # Output: (batch, 4)
        return x
