import torch
from torch.nn.functional import mse_loss
from torchmetrics.functional import r2_score
import numpy as np
class model():
    def __init__(self, m, train_data, test_data, iterations=200):
        self.m = m
        self.train_data = train_data
        self.test_data = test_data
        self.iterations = iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.m.to(self.device)

    def train(self):
        optimizer = torch.optim.Adam(self.m.parameters(), lr=0.01)
        loss = 0
        self.m.train()
        for epoch in range(self.iterations):
            print(epoch, loss)
            for batch in self.train_data:
                data, labels = batch['data'].to(self.device), batch['label'].to(self.device)
                optimizer.zero_grad()
                predictions = self.m(data)
                loss = torch.nn.functional.mse_loss(predictions, labels)
                loss.backward()
                optimizer.step()
        
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