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

    dataset = data_io.dataloader.RawDataset(root_dir='./raw_data')
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    for batch in dataloader:
        data, labels = batch['data'], batch['label']
        print(data.shape, labels.shape)
    
    m = analysis.models.EEGCNN(filter_size=3)
    m(data)
#tensor = torch.from_numpy(coh).float()
#m(tensor.unsqueeze(0).unsqueeze(0).squeeze(-1)) # [1, 1, 19, 19])
# how bad would it be to pilot the main program with a numpy array, and when using 
# feature selection or visualsation
#visualization.plot.raw_plot(testEpoch)
#visualization.plot.raw_sensors(testEpoch)