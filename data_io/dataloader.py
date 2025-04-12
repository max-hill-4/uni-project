import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import numpy as np
from pathlib import Path
import csv 
import pandas 
from features_io import features
from data_io import raw
class RawDataset(Dataset):
    """
    Custom Dataset for loading .mat files
    
    Args:
        root_dir (str): Directory containing .mat files
        transform (callable, optional): Optional transform to be applied
    """
    def __init__(self, root_dir, transform=None, feature='coh'):
        self.root_dir = Path(root_dir)
        self.mat_files = list(self.root_dir.glob('*.mat'))  # Get all .mat files
        self.transform = transform
        self.feature = feature
        
    def __len__(self):
        return len(self.mat_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        eeg_data = loadmat(self.mat_files[idx])
        df = pandas.read_csv(r'./raw_data/biomarkers.csv')
        
        if self.feature == 'coh':
            eeg_data = features.Coherance.coh(eeg_data)
        
        # df -> feature 19 x 19 ? 
        
        label_data = (df[df['Participant']=='A']).to_numpy()
        label_data = label_data[0][2:]
        label_data = np.asarray(label_data, dtype=np.float64)

        sample = {
            'data': eeg_data,  # this is a numpy array of epoch idx.
            'label': label_data  
        }
        # Convert to torch tensors
        sample['data'] = torch.from_numpy(sample['data']).float()
        sample['label'] = torch.from_numpy(sample['label']) # or .float() for regression

        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Example usage:
if __name__ == '__main__':

    dataset = RawDataset(root_dir='./raw_data')
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    for batch in dataloader:
        data, labels = batch['data'], batch['label']
        print(data.shape, labels.shape)