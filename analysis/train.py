import torch
from torch.nn.functional import mse_loss
from torchmetrics.functional import r2_score
class model():
    def __init__(self, m, train_data, test_data, iterations=10):
        self.m = m
        self.train_data = train_data
        self.test_data = test_data
        self.iterations = iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.m.to(self.device)


    def train(self):
        # Optimizer and loss function
        optimizer = torch.optim.Adam(self.m.parameters(), lr=0.001)
        loss_fn = torch.nn.SmoothL1Loss()
        
        all_predictions = []
        all_labels = []

        for epoch in range(self.iterations):
            epoch_loss = 0  # To track loss for each epoch
            
            for batch in self.train_data:
                data, labels = batch['data'].to(self.device), batch['label'].to(self.device)
                
                optimizer.zero_grad()  # Zero out gradients

                # Forward pass
                predictions = self.m(data)
                all_predictions.append(predictions.cpu())  # Store predictions for evaluation later
                all_labels.append(labels.cpu())  # Store labels for evaluation
                
                # Compute loss
                loss = loss_fn(predictions, labels)
                epoch_loss += loss.item()  # Accumulate loss for reporting
                
                # Backward pass and optimization
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.m.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()

            # Step the scheduler at the end of each epoch


    def predict(self):
        all_predictions = []
        all_labels = []
        self.m.eval()        
        with torch.no_grad():  # Disable gradient calculations for inference
            for batch in self.test_data:
                data, labels = batch['data'].to(self.device), batch['label'].to(self.device)  # Move data and labels to the device

                # Get predictions from the model
                predictions = self.m(data)
                # Store predictions and labels
                all_predictions.append(predictions.cpu())  # Store predictions on the CPU for evaluation later
                all_labels.append(labels.cpu())  # Store labels on the CPU
                print
        # After collecting predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)  # Ensure proper concatenation along the batch dimension
        all_labels = torch.cat(all_labels, dim=0)
        return all_predictions, all_labels
    
    def mse(self, predict, truth):
        mse = mse_loss(predict, truth)
        return mse
    
    def mse_per_class(self, predictions, labels):
        # Ensure inputs have the same shape and are 2D
        assert predictions.shape == labels.shape and len(predictions.shape) == 2, "Inputs must be 2D tensors of same shape"
        
        # Get the number of classes (second dimension)
        num_classes = predictions.shape[1]
        
        # Compute squared differences
        squared_diff = (predictions - labels) ** 2
        
        # Compute MSE per class (mean along dimension 0, i.e., across the 62 samples)
        mse_per_class = torch.mean(squared_diff, dim=0)
        
        # Return as a dictionary mapping class index to MSE
        return {i: mse_per_class[i].item() for i in range(num_classes)}
    
    def r2_per_class(self, predictions: torch.Tensor, labels: torch.Tensor) -> dict:

        assert predictions.shape == labels.shape and len(predictions.shape) == 2, \
            "Inputs must be 2D tensors of same shape"

        num_classes = predictions.shape[1]
        r2_values = r2_score(predictions, labels, multioutput='raw_values')
        return {i: r2_values[i].item() for i in range(num_classes)} 
    

import torch
import torch.nn as nn
import torchvision.models as models

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
