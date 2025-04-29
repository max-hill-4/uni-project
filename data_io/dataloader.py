import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
from pathlib import Path
import pandas 
from torch.utils.data import Subset
from features_io import FeatureExtractor

class RawDataset(Dataset):

    def __init__(self, sleep_stages, feature_freq, hormones):
        self.root_dir = Path(r'/mnt/block/eeg')
        self.mat_files = [
            mat_file
            for stage in sleep_stages
            for mat_file in (self.root_dir / stage).glob('*.mat')
        ]
        self.feature_extractor = FeatureExtractor(feature_freq)
        self.bdc_columns = hormones
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
            'label': torch.tensor(label_data, dtype=torch.float32),
        }

        return sample

    def _load_labels(self, filepath = (r'./raw_data/biomarkers.csv'), scaler = 'minmax'):
        "Needs to return dictionary of shape {particpant: {BDC : [nparray of shape [12]], ...}, ...}"   
        labels = {}
        df = pandas.read_csv(r'./raw_data/biomarkers.csv').dropna()

        scaler = MinMaxScaler()
        # Might be a good idea to not scale to complete 0 and 1 ?
        # feature_range=(-1, 1)

        # Scale the hormone columns in the DataFrame
        df[self.bdc_columns] = scaler.fit_transform(df[self.bdc_columns])

        labels = {}
        participants = df['Participant'].unique()
        for participant in participants:
            values = []
            participant_df = df[df['Participant'] == participant]
            for hormone in self.bdc_columns:
                hormone_value = participant_df[hormone].to_numpy().flatten()[0]
                values.append(hormone_value)
            labels[participant] = values
            
        return labels

def participant_split(dataset, train_proportion):


    # Get unique participants with valid labels
    participants = list(dataset.labels.keys())
    num_participants = len(participants)
    
    # Use torch.randperm to shuffle participant indices
    torch.manual_seed(42)
    indices = torch.randperm(num_participants)
    
    # Calculate number of training participants
    num_train = int(train_proportion * num_participants)
    
    # Split participants
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_participants = [participants[i] for i in train_indices]
    test_participants = [participants[i] for i in test_indices]
    print(f'Traning Parps{train_participants}, Testing Parps {test_participants}')
    train_sample_indices = [
        idx for idx, participant in enumerate(dataset.mat_files)
        if str(participant)[24] in train_participants
    ]
    test_sample_indices = [
        idx for idx, participant in enumerate(dataset.mat_files)
        if str(participant)[24] in test_participants
    ]
    train_subset = Subset(dataset, train_sample_indices)
    test_subset = Subset(dataset, test_sample_indices)

    return train_subset, test_subset