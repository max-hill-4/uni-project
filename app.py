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
        mse_results[te_parps[0]] = l
        r2_results[te_parps[0]] = e
        print(f'trnaing parps are : {tr_parps}') 
        print(mse_results , "\n")
        print(r2_results)

        if e[0] > 0.1:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.scatter(p[0],p[1] , color='blue', alpha=0.5, label='Predicted vs Truth') 
            plt.savefig(f'{e}.png', format='png', dpi=300, bbox_inches='tight')
            save(m.state_dict(), f'./trained_models/{args}.model.pt')
        
            with open('results.json', 'a') as f:
                json.dump({int(time.time()) : { 'args' : args, 'mse' : mse_results, 'r2' : r2_results}}, f, indent=4)


if __name__ == '__main__':

    from itertools import product
    params = {
        "b_size" : 4 ,
        "filter_size" : 3,
        "iterations" : 5,
        "k_folds" : 5,
        "in_channels" : 1
    }

    param_options = {
        "feature_freq" : [[{'coh' : 'alpha'}, {'coh' : 'beta'}], [{'coh' : 'alpha'}]],
        "hormones" : [['BDC1.2']],
        "sleep_stages" : [['N1']],
    }

    for feature_freq, hormone, sleep_stage in product(
        param_options["feature_freq"],
        param_options["hormones"],
        param_options["sleep_stages"]
    ):
        params = params.copy()
        params.update({
            "feature_freq": feature_freq,
            "hormones": hormone,
            "sleep_stages": sleep_stage
        })
        
        print(f"Running with params: {params}")  # Optional: log the params
        if __name__ == "__main__":
            main(**params)
