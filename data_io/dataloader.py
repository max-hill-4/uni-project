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
        self.transform = transform
        self.feature = feature
        self.df = pandas.read_csv(r'./raw_data/biomarkers.csv').dropna()
        self.bdc_columns = ['BDC1'] + [f'BDC1.{i}' for i in range(1, 12)]

        
    def __len__(self):
        return len(self.mat_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = self.mat_files[idx]
        participant = str(file)[24]
        eeg_data = loadmat(file)
        
        if self.feature == 'coh':
            eeg_data = features.Coherance.coh(eeg_data)
            
        if self.feature == 'sl':
            eeg_data = features.SynchronizationLikelihood.compute_synchronization_likelihood(eeg_data)
        

        label_data = self.df.loc[self.df['Participant'] == participant, self.bdc_columns].to_numpy()
        label_data = np.asarray(label_data, dtype=np.float64).squeeze()
        if label_data.size == 0:
            #print(f'WARNING: {participant} has no labels!')
            return None
        sample = {
            'data': torch.tensor(eeg_data, dtype=torch.float32),
            'label': torch.tensor(label_data, dtype=torch.float32)  # Directly convert
        }

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