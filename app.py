import analysis.models
import analysis.predict
import data_io
import data_io.dataloader
import visualization
import os
import analysis
import features_io
import numpy as np
import torch
from torch.utils.data import DataLoader
if __name__ == '__main__':

    dataset = data_io.dataloader.RawDataset(root_dir='./raw_data') # could pass params like, hormone type, feature type etc etc. 
    
    train_data = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    m = analysis.models.EEGCNN(filter_size=3, num_classes=108)

    a = analysis.train.model(m, train_data, train_data, iterations=10)

    a.train()
    p = a.predict()
    e = a.mse(*p)
    print(f"MSE: {e.item():.4f}")
