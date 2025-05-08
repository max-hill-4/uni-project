import analysis.models
import data_io.dataloader
import data_io.sampler
import data_io.writeresults
from torch.utils.data import DataLoader
from params import param_options
from itertools import product

def main(**args):

    dataset = data_io.dataloader.RawDataset(sleep_stages=args["sleep_stages"], feature_freq=args["feature_freq"], hormones = args["hormones"]) 
    folds = data_io.sampler.participant_kfold_split(dataset, args["k_folds"])
    results = []
    for fold in folds:

        train_dataset, test_dataset, tr_parps, te_parps = fold
        print(len(train_dataset))
        print(len(test_dataset))
        print(f'TRANING PARSP{tr_parps}')
        print(f'TESTING PARPS{te_parps}')
        train_data = DataLoader(train_dataset, batch_size=args["b_size"], shuffle=True, num_workers=4, collate_fn=data_io.dataloader.collate_fn)    
        test_data = DataLoader(test_dataset, batch_size=args["b_size"], shuffle=True, num_workers=4, collate_fn=data_io.dataloader.collate_fn)
        print(len(train_data))
        print(len(test_data))
        m = analysis.models.EEGCNN(filter_size=args["filter_size"], num_classes=len(args['hormones']), in_channels=args["in_channels"])
        a = analysis.train.model(m, train_data, test_data, iterations=args[ "iterations" ])

        a.train()

        predictions, truth = a.predict()

        results.append(a.accuracy(predictions, truth))
    
    data_io.dataloader.write_results(args, results)


if __name__ == '__main__':
    import os 
    print(os.getcwd())
    params = {
        "b_size" : 16,
        "filter_size" : 5,
        "iterations" : 50,
        "k_folds" : 3,
        "in_channels" : 1
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
            import os
            print(os.getcwd())
            main(**params)