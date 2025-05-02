import data_io.dataloader
import analysis
from torch.utils.data import DataLoader
from torch import save
import json
import time
def main(**args):


    dataset = data_io.dataloader.RawDataset(sleep_stages=args["sleep_stages"], feature_freq=args["feature_freq"], hormones = args["hormones"]) 

    folds = data_io.dataloader.participant_kfold_split(dataset, args["k_folds"])
    mse_results = {}
    r2_results = {}
    
    for fold in folds:
        train_dataset, test_dataset, tr_parps, te_parps = fold
        train_data = DataLoader(train_dataset, batch_size=args["b_size"], shuffle=True, num_workers=4, collate_fn=data_io.dataloader.collate_fn)    
        test_data = DataLoader(test_dataset, batch_size=args["b_size"], shuffle=True, num_workers=4, collate_fn=data_io.dataloader.collate_fn)
        
        m = analysis.models.EEGCNN(filter_size=args["filter_size"], num_classes=len(args['hormones']), in_channels=args["in_channels"])

        a = analysis.train.model(m, train_data, test_data, iterations=args[ "iterations" ])
        a.train()

        p = a.predict()
        l = a.mse_per_class(*p)
        e = a.r2_per_class(*p)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.scatter(p[0],p[1] , color='blue', alpha=0.5, label='Predicted vs Truth') 
        plt.savefig(f'{e}.png', format='png', dpi=300, bbox_inches='tight')
        mse_results[te_parps[0]] = l
        r2_results[te_parps[0]] = e
        print(f'trnaing parps are : {tr_parps}') 
        print(mse_results , "\n")
        print(r2_results)
    
        save(m.state_dict(), f'./trained_models/{args}.model.pt')
        
        with open('results.json', 'a') as f:
            json.dump({int(time.time()) : { 'args' : args, 'mse' : mse_results, 'r2' : r2_results}}, f, indent=4)


if __name__ == '__main__':

    params = {
        "feature_freq" : [{'coh' : 'alpha'}, {'coh' : 'beta'}],
        "hormones" : ['BDC1.2'],
        "sleep_stages" : ['N1'],
        "b_size" : 4 ,
        "filter_size" : 3,
        "iterations" : 5,
        "k_folds" : 11,
        "in_channels" : 1
    }

    main(**params)