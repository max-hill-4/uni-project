import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import numpy as np
from pathlib import Path
import pandas 
from features_io import features
class RawDataset(Dataset):

    def __init__(self, sleep_stages, transform=None, feature='coh'):
        self.root_dir = Path(r'/mnt/block/eeg')
        self.mat_files = [
            mat_file
            for stage in sleep_stages
            for mat_file in (self.root_dir / stage).glob('*.mat')
        ]
        print(self.mat_files)
        self.transform = transform
        self.feature = feature
        self.df = pandas.read_csv(r'./raw_data/biomarkers.csv')

        
    def __len__(self):
        return len(self.mat_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = self.mat_files[idx]
        participant = str(file)[15]
        eeg_data = loadmat(file)
        
        if self.feature == 'coh':
            eeg_data = features.Coherance.coh(eeg_data)
            
        if self.feature == 'sl':
            eeg_data = features.SynchronizationLikelihood.compute_synchronization_likelihood(eeg_data)
        
        label_data = (self.df[self.df['Participant']==participant]).to_numpy()
        label_data = label_data[0][2:]
        label_data = np.asarray(label_data, dtype=np.float64)
        sample = {
            'data': eeg_data,  # this is a numpy array of epoch idx.
            'label': label_data  
        }
        # Convert to torch tensors
        sample['data'] = torch.from_numpy(sample['data']).float()
        sample['label'] = torch.from_numpy(sample['label']).float() # or .float() for regression
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