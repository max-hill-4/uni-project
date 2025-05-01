import data_io.dataloader
import analysis
from torch.utils.data import DataLoader
if __name__ == '__main__':

    feature_freq = [{'sl' : 'alpha'}]
    hormones = ['BDC1.1']
    sleep_stages = ["N1"]
    b_size = 64 
    filter_size = 5
    iterations = 3
    k_folds = 7
    in_channels = 1

    dataset = data_io.dataloader.RawDataset(sleep_stages=sleep_stages, feature_freq=feature_freq, hormones = hormones) 

    folds = data_io.dataloader.participant_kfold_split(dataset, k_folds)
    mse_results = {}
    r2_results = {}
    
    for fold in folds:
        train_dataset, test_dataset, tr_parps, te_parps = fold
        train_data = DataLoader(train_dataset, batch_size=b_size, shuffle=True, num_workers=4, collate_fn=data_io.dataloader.collate_fn)    
        test_data = DataLoader(test_dataset, batch_size=b_size, shuffle=True, num_workers=4, collate_fn=data_io.dataloader.collate_fn)
        
        m = analysis.models.EEGCNN(filter_size=filter_size, num_classes=1, in_channels=in_channels)

        a = analysis.train.model(m, train_data, test_data, iterations=iterations)
        a.train()

        p = a.predict()
        l = a.mse_per_class(*p)
        e = a.r2_per_class(*p)
        
        mse_results[te_parps[0]] = l
        r2_results[te_parps[0]] = e
        
        print(mse_results , "\n")
        print(r2_results)
