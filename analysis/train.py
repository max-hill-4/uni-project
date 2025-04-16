import torch
from torch.nn.functional import mse_loss
from sklearn.metrics import confusion_matrix
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

        cf_matrix = confusion_matrix(all_labels, all_predictions)
        all_predictions = torch.cat(all_predictions, dim=0)  # Ensure proper concatenation along the batch dimension
        all_labels = torch.cat(all_labels, dim=0)
        return all_predictions, all_labels
    
    def mse(self, predict, truth):
        
        cf_matrix = confusion_matrix(predict, truth)
        all_predictions = torch.cat(all_predictions, dim=0)  # Ensure proper concatenation along the batch dimension
        all_labels = torch.cat(all_labels, dim=0)
        mse = mse_loss(predict, truth)
        print(mse)
        