import data_io
import data_io.dataloader
import analysis
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
import torch
if __name__ == '__main__':

    def custom_collate_fn(batch):
        # Filter out None samples
        batch = [sample for sample in batch if sample is not None]
        
        if not batch:
            return None  # or return empty tensors depending on your training logic

        # Stack the data and labels
        data = torch.stack([sample['data'] for sample in batch])
        labels = torch.stack([sample['label'] for sample in batch])
        
        return {'data': data, 'label': labels}

 

    dataset = data_io.dataloader.RawDataset(sleep_stages=["N1"], feature='coh') 
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) 
    
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    test_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    print("finished loading data!")
    m = analysis.models.EEGCNN(filter_size=3, num_classes=12)

    a = analysis.train.model(m, train_data, test_data, iterations=5)

    a.train()
    p = a.predict()
    l = a.mse_per_class(*p)
    e = a.r2_per_class(*p)
    print(e)