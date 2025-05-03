import torch.nn as nn
import torchvision.models as models

class EEGCNN(nn.Module):
    def __init__(self, in_channels: int = 1, filter_size: int = 3, num_classes: int = 108):
        """
        Args:
            in_channels: Number of input channels (e.g., 1 for single-channel, 5 for multi-channel).
            filter_size: 3 (C3), 6 (C6), or 8 (C8).
            num_classes: Number of output classes.
        """
        super().__init__()
        
        # Convolutional layers (now supports multi-channel input
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=filter_size, padding=filter_size // 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=filter_size, padding=filter_size // 2),
            nn.ReLU(),
            #nn.MaxPool2d(2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 19 * 19, 256),  # Increased capacity
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        # Input shape: (batch, in_channels, 19, 19)
        x = self.conv(x)            # Output: (batch, 32, 9, 9)
        x = self.classifier(x)      # Output: (batch, num_classes)
        return x

class ResNetEEG(nn.Module):
    def __init__(self, in_channels: int = 1, filter_size: int = 3, num_classes: int = 108):
        """
        Args:
            in_channels: Number of input channels (e.g., 1 for single-channel, 5 for multi-channel).
            filter_size: Size of the filter (typically 3x3, 7x7, etc.).
            num_classes: Number of output classes.
        """
        super().__init__()

        # Initialize the pre-trained ResNet model (ResNet-18 for simplicity)
        resnet = models.resnet18(pretrained=False)  # Load ResNet-18 architecture (without pre-trained weights)

        # Modify the first convolution layer to accept in_channels (e.g., 1 or 5 channels for EEG)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=filter_size, stride=2, padding=filter_size // 2, bias=False)

        # Optionally, modify the fully connected layer to match the number of output classes
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        # To match your desired format, we'll keep the residual layers as they are in ResNet, but use the classifier format
        self.resnet = resnet

    def forward(self, x):
        """
        Forward pass through the network
        """
        # Input shape: (batch, in_channels, 19, 19)
        x = self.resnet(x)  # Output: (batch, num_classes)
        return x
    
