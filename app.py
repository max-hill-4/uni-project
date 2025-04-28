import data_io
import data_io.dataloader
import analysis
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
import torch
if __name__ == '__main__':

    def collate_fn(batch):
        filtered_batch = {'data': [], 'label': []}

        for sample in batch:
            if sample is not None:
                filtered_batch['data'].append(sample['data'])
                filtered_batch['label'].append(sample['label'])
            
        data = torch.stack(filtered_batch['data'])
        labels = torch.stack(filtered_batch['label'])
        return {'data': data, 'label': labels}

 

    dataset = data_io.dataloader.RawDataset(sleep_stages=["N1"], feature='coh', hormones=['BDC1.4',]) 
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) 
    
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    m = analysis.models.EEGCNN(filter_size=3, num_classes=1)

    a = analysis.train.model(m, train_data, test_data, iterations=5)

    a.train()
    p = a.predict()
    l = a.mse_per_class(*p)
    e = a.r2_per_class(*p)
    print(e)