import torch
from torch.nn.functional import mse_loss
from torchmetrics.functional import r2_score
import torch.nn as nn
import copy

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
        optimizer = torch.optim.Adam(self.m.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss() 
        patience = 0 
        all_predictions = []
        all_labels = []
        b_val_ac = 0
        losses = []  # List to store average loss per epoch
        results = []
        for epoch in range(self.iterations):
            epoch_loss = 0.0  # Accumulate loss for the epoch
            num_batches = 0
            
            for batch in self.train_data:
                data, labels = batch['data'].to(self.device), batch['label'].to(self.device)
                labels = labels.squeeze(1)
                optimizer.zero_grad()  # Zero out gradients
                # Forward pass
                predictions = self.m(data)
                all_predictions.append(predictions.cpu())  # Store predictions
                all_labels.append(labels.cpu())  # Store labels
     
                results.append(self.accuracy(predictions, labels))
                # Compute loss
                loss = criterion(predictions, labels)
                epoch_loss += loss.item()  # Accumulate loss
                num_batches += 1
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
             

            traning_acc = sum(results) / len(results)
            val_acc = self.accuracy(*self.predict())

            if val_acc > b_val_ac:
                b_val_ac = val_acc
                best_model_state = copy.deepcopy(self.m.state_dict())
                patience = 0
            else :
                patience+=1
            # Compute average loss for the epoch
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{self.iterations}, Average Loss: {avg_loss:.4f}, Traning Acc: {traning_acc} Val Acc: {val_acc}")
            if patience >= 10:
                ('No improvmenet in val acc')
                self.m.load_state_dict(best_model_state)
                break

        return losses, all_predictions, all_labels


    def predict(self):
        all_predictions = []
        all_labels = []
        self.m.eval()
        with torch.no_grad():  # Disable gradient calculations for inference
            for batch in self.test_data:
                data, labels = batch['data'].to(self.device), batch['label'].to(self.device)  # Move data and labels to the device

                labels = labels.squeeze(1)
                # Get predictions from the model
                predictions = self.m(data)  # Shape: (batch_size, 3) - raw logits
                # Store predictions and labels
                all_predictions.append(predictions.cpu())  # Store logits on CPU
                all_labels.append(labels.cpu())  # Store labels on CPU

        # Concatenate predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)  # Shape: (n_samples, 3)
        all_labels = torch.cat(all_labels, dim=0)  # Shape: (n_samples,)
        # Apply softmax to convert logits to probabilities
        all_probabilities = torch.softmax(all_predictions, dim=1)  # Shape: (n_samples, 3)
        # Map indices to class names (optional)
        # Return data, probabilities, predicted classes, and labels
        return all_probabilities, all_labels
    
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
    
    def accuracy(self, predictions: torch.Tensor, labels: torch.Tensor):
        predicted_classes = torch.argmax(predictions, dim=1)
        correct = (predicted_classes == labels).sum().item()
        accuracy = correct / len(labels)
        return accuracy