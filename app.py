import analysis.models
import data_io
import data_io.dataloader
import visualization
import os
import analysis
import features_io
import numpy as np
import torch
from torch.utils.data import DataLoader
#testEpoch = data_io.raw.epochtoRawArray(r'./raw_data/bdc14_A1_0026.mat')
#coh = features_io.features.Coherance.coh(testEpoch)

if __name__ == '__main__':

    dataset = data_io.dataloader.RawDataset(root_dir='./raw_data') # could pass params like, hormone type, feature type etc etc. 
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    m = analysis.models.EEGCNN(filter_size=3, num_classes=108)
    m.eval()

    # Optional: If you have a trained model, load its weights
    # m.load_state_dict(torch.load('path_to_saved_model.pth'))
    
    # Move the model and data to the same device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
    # Predict on unseen data
    all_predictions = []
    all_labels = []
    total_loss = 0
    m.train()
    for epoch in range(200):
        print(epoch)
        for batch in dataloader:
            data, labels = batch['data'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            predictions = m(data)
            loss = torch.nn.functional.mse_loss(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    m.eval()        
    with torch.no_grad():  # Disable gradient calculations for inference
        for batch in dataloader:
            data, labels = batch['data'].to(device), batch['label'].to(device)  # Move data and labels to the device
            
            # Get predictions from the model
            predictions = m(data)
            
            # Store predictions and labels
            all_predictions.append(predictions.cpu())  # Store predictions on the CPU for evaluation later
            all_labels.append(labels.cpu())  # Store labels on the CPU

    # After collecting predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)  # Ensure proper concatenation along the batch dimension
    all_labels = torch.cat(all_labels, dim=0)
    for batch in range(3):
        print(f"THIS IS BATCH N.{batch}")
        for i in range(108):
            print(f"predicted val = {all_predictions[batch][i]} Truth Val = {all_labels[batch][i]}")
    from torch.nn.functional import mse_loss

    # Calculate MSE
    mse = mse_loss(all_predictions, all_labels)
    print(f"MSE: {mse.item():.4f}")
    print(all_predictions.shape, all_labels.shape)