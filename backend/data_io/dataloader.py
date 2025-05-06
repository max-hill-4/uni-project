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
        
        participant = str(file)[18]
        if participant in self.labels:
            label_data = self.labels[participant]
        else:
            return None
        
        sample = {
            'data': torch.tensor(eeg_data, dtype=torch.float32),
            'label': torch.tensor(label_data, dtype=torch.long),
        }
        return sample

    def _load_labels(self, scaler = 'minmax'):
        "Needs to return dictionary of shape {particpant: {BDC : [nparray of shape [12]], ...}, ...}"   
        labels = {}
        df = pandas.read_csv(r'/mnt/raw_data/biomarkers_quartiles.csv').dropna()

        labels = {}
        participants = df['Participant'].unique()
        for participant in participants:
            values = []
            participant_df = df[df['Participant'] == participant]
            for h in self.hormones:
                hormone_value = participant_df[h].values[0]
                values.append(hormone_value)
            labels[participant] = values
        return labels

def participant_kfold_split(dataset, n_splits=5, shuffle=True, random_state=None):
    """
    Returns K folds with participant-wise separation
    
    Args:
        dataset: Your dataset object (must have labels/mat_files accessible)
        n_splits: Number of folds (default: 5)
        shuffle: Whether to shuffle participants (default: True)
        random_state: Seed for reproducibility (default: None)
        
    Returns:
        List of K folds, each containing (train_subset, test_subset)
    """
    torch.manual_seed(42)
    # Get unique participants
    participants = list(dataset.labels.keys())
    num_participants = len(participants)
    # Shuffle participants if needed
    if n_splits == 1:
        train_indices = [
            idx for idx, p in enumerate(dataset.mat_files) 
            if str(p)[18] in participants
        ]
        return [(
            Subset(dataset, train_indices),
            Subset(dataset, []),  # Empty test set
            participants,         # All participants in train
            []                   # No test participants
        )]


    if shuffle:
        if random_state is not None:
            torch.manual_seed(random_state)
        indices = torch.randperm(num_participants).tolist()
        participants = [participants[i] for i in indices]
    # Create folds
    folds = []
    for fold in range(n_splits):
        # Calculate test participant range for this fold
        fold_size = num_participants // n_splits
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold != n_splits - 1 else num_participants
        
        # Split participants
        test_participants = participants[test_start:test_end]
        train_participants = [p for p in participants if p not in test_participants]
        
        # Get sample indices
        train_indices = [
            idx for idx, p in enumerate(dataset.mat_files) 
            if str(p)[18] in train_participants
        ]
        test_indices = [
            idx for idx, p in enumerate(dataset.mat_files) 
            if str(p)[18] in test_participants
        ]
        
        folds.append((
            Subset(dataset, train_indices),
            Subset(dataset, test_indices),
            train_participants,  # Optional: return participant lists
            test_participants    # for tracking
        ))

    return folds

def collate_fn(batch):
    filtered_batch = {'data': [], 'label': []}

    for sample in batch:
        if sample is not None:
            filtered_batch['data'].append(sample['data'])
            filtered_batch['label'].append(sample['label'])
        
    data = torch.stack(filtered_batch['data'])
    labels = torch.stack(filtered_batch['label'])
    return {'data': data, 'label': labels}