import data_io
import data_io.dataloader
import analysis
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
import torch

def collate_fn(batch):
    filtered_batch = {'data': [], 'label': []}

    for sample in batch:
        if sample is not None:
            filtered_batch['data'].append(sample['data'])
            filtered_batch['label'].append(sample['label'])
        
    data = torch.stack(filtered_batch['data'])
    labels = torch.stack(filtered_batch['label'])
    return {'data': data, 'label': labels}

if __name__ == '__main__':

    dataset = data_io.dataloader.RawDataset(sleep_stages=["N2"], feature_name='coh', hormones=['BDC1', 'BDC1.1', 'BDC1.2', 'BDC1.3']) 

    train_dataset, test_dataset = data_io.dataloader.participant_split(dataset, 0.8) 
    
    train_data = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)    
    test_data = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    m = analysis.models.EEGCNN(filter_size=3, num_classes=4)

    a = analysis.train.model(m, train_data, test_data, iterations=5)

    a.train()
    p = a.predict()
    l = a.mse_per_class(*p)
    e = a.r2_per_class(*p)
    print(e)