import analysis.models
import data_io
from torch.utils.data import DataLoader
import data_io
from params import param_options
from itertools import product
from torch import save 
def main(**args):

    dataset = data_io.dataloader.RawDataset(sleep_stages=args["sleep_stages"], feature_freq=args["feature_freq"], hormones = args["hormones"]) 
    folds = data_io.sampler.participant_kfold_split(dataset, args["k_folds"])
    results = []
    for fold in folds:

        train_dataset, test_dataset, tr_parps, te_parps = fold
        train_data = DataLoader(train_dataset, batch_size=args["b_size"], shuffle=True, num_workers=4, collate_fn=data_io.dataloader.collate_fn)    
        test_data = DataLoader(test_dataset, batch_size=args["b_size"], shuffle=True, num_workers=4, collate_fn=data_io.dataloader.collate_fn)
        m = analysis.models.EEGCNN(filter_size=args["filter_size"], num_classes=len(args['hormones']), in_channels=args["in_channels"])
        a = analysis.train.model(m, train_data, test_data, iterations=args[ "iterations" ])

        a.train()

        data, predictions, truth = a.predict()
        accuracy = a.accuracy(predictions, truth) 
        if accuracy > max(results):
           save(m.state_dict(), f'/trained_models/classification/{args[feature_freq]}{args[hormone]}.pth') 
        results.append(accuracy)

    data_io.writeresults.write_results(args, results)


if __name__ == '__main__':
    import os 
    print(os.getcwd())
    params = {
        "b_size" : 16,
        "filter_size" : 5,
        "iterations" : 30,
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