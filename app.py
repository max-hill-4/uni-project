import data_io
import data_io.dataloader
import analysis
from torch.utils.data import DataLoader, random_split
if __name__ == '__main__':

 
    dataset = data_io.dataloader.RawDataset(stages=("N1"), feature='coh') # could pass params like, hormone type, feature type etc etc. 
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) 
    
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_data = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    m = analysis.models.EEGCNN(filter_size=3, num_classes=108)

    a = analysis.train.model(m, train_data, test_data, iterations=5)

    a.train()
    p = a.predict()
    e = a.mse(*p)
    print(f"MSE: {e.item():.4f}")
