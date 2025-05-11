import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
from pathlib import Path
import pandas 
from torch.utils.data import Subset
from features_extract import FeatureExtractor

class RawDataset(Dataset):

    def __init__(self, sleep_stages, feature_freq, hormones):
        self.root_dir = Path(r'/mnt/eeg')
        self.mat_files = [
            mat_file
            for stage in sleep_stages
            for mat_file in (self.root_dir / stage).glob('*.mat')
        ]

        self.feature_extractor = FeatureExtractor(feature_freq)
        self.hormones = hormones
        self.labels = self._load_labels()

    def __len__(self):
        return len(self.mat_files)
    
    def __getitem__(self, idx):

        file = self.mat_files[idx]
        eeg_data = loadmat(file)
        
        eeg_data = self.feature_extractor.get(eeg_data) 
        
        participant = str(file)[24]
        if participant in self.labels:
            label_data = self.labels[participant]
        else:
            return None
        
        sample = {
            'data': torch.tensor(eeg_data, dtype=torch.float32),
            'label': torch.tensor(label_data, dtype=torch.long),
        }
        print(sample['data'])
        return sample

    def _load_labels(self, scaler = 'minmax'):
        "Needs to return dictionary of shape {particpant: {BDC : [nparray of shape [12]], ...}, ...}"   
        labels = {}
        df = pandas.read_csv(r'/mnt/raw_data/biomarkers_thirds.csv').dropna()

        labels = {}
        participants = df['Participant'].unique()
        for participant in participants:
            values = []
            participant_df = df[df['Participant'] == participant]
            for h in self.hormones:
                hormone_value = participant_df[h].values[0]
                values.append(hormone_value)
            labels[participant] = values
        print(labels)
        return labels



def collate_fn(batch):
    filtered_batch = {'data': [], 'label': []}

    for sample in batch:
        if sample is not None:
            filtered_batch['data'].append(sample['data'])
            filtered_batch['label'].append(sample['label'])
        
    data = torch.stack(filtered_batch['data'])
    labels = torch.stack(filtered_batch['label'])
    return {'data': data, 'label': labels}