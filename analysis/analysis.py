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
        self.conv = nn.Conv2d(
            in_channels=1,           # Input: 1 channel (19×19 matrix)
            out_channels=1,          # Single filter
            kernel_size=filter_size, # Square filter (3×3, 6×6, or 8×8)
            padding=filter_size // 2 # Preserves 19×19 output
        )
        
        # Fully connected layer (input size = 19×19 = 361)
        self.fc = nn.Linear(19 * 19, num_classes)  # Output: 4 classes

    def forward(self, x):
        # Input shape: (batch, 1, 19, 19)
        x = self.conv(x)            # Still (batch, 1, 19, 19)
        x = x.view(x.size(0), -1)   # Flatten to (batch, 361)
        x = self.fc(x)              # Output: (batch, 4)
        return x

# Variants (C3, C6, C8)
class C3(EEGCNN):
    def __init__(self, num_classes=4): super().__init__(3, num_classes)
class C6(EEGCNN):
    def __init__(self, num_classes=4): super().__init__(6, num_classes)
class C8(EEGCNN):
    def __init__(self, num_classes=4): super().__init__(8, num_classes)

# Example Usage
if __name__ == "__main__":
    # 1. Initialize models
    c3, c6, c8 = C3(), C6(), C8()

    # 2. Test with random 19×19 input (batch_size=5)
    dummy_input = torch.randn(5, 1, 19, 19)  # Shape: (batch, channel, height, width)
    
    # 3. Forward pass
    for model in [c3, c6, c8]:
        output = model(dummy_input)
        print(f"{model.__class__.__name__} output shape: {output.shape}")  # Should be (5, 4)