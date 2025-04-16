import data_io
import data_io.dataloader
import analysis
from torch.utils.data import DataLoader
if __name__ == '__main__':

    train_dataset = data_io.dataloader.RawDataset(root_dir='./raw_data', feature='coh') # could pass params like, hormone type, feature type etc etc. 
    test_dataset = data_io.dataloader.RawDataset(root_dir='./raw_data/test', feature='sl')
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    m = analysis.models.EEGCNN(filter_size=3, num_classes=108)

    a = analysis.train.model(m, train_data, test_data, iterations=100)

    a.train()
    p = a.predict()
    e = a.mse(*p)
    print(f"MSE: {e.item():.4f}")
