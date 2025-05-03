import analysis.models
import data_io.dataloader
import analysis
from torch.utils.data import DataLoader
from torch import save
import json
import time
import matplotlib.pyplot as plt
from tests import param_options
import torch
from itertools import product
def compute_saliency_map(model, input_tensor, device):
    model.eval()
    input_tensor = input_tensor.to(device).requires_grad_(True)
    output = model(input_tensor)  # Shape: [1, 1]
    output.backward()  # Compute gradients w.r.t. input
    saliency = input_tensor.grad.abs().squeeze()  # Shape: [1, 19, 19]
    return saliency.cpu().numpy()

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

        all_inputs, predictions, truth = a.predict()
        input_tensor = all_inputs[0:1]

        l = a.mse_per_class(predictions, truth)
        e = a.r2_per_class(predictions, truth)
        mse_results[te_parps[0]] = l
        r2_results[te_parps[0]] = e
        print(f'trnaing parps are : {tr_parps}') 
        print(mse_results , "\n")
        print(r2_results)
        import torch


        if e[0] > 0:
            saliency_map = compute_saliency_map(m, input_tensor, device)

            plt.imshow(saliency_map, cmap="jet", alpha=0.5)
            plt.colorbar()
            plt.savefig(f'./backend/intrestingdata/saliency/{e[0]}.png', dpi=300)
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.scatter(predictions, truth , color='blue', alpha=0.5, label='Predicted vs Truth') 
            plt.savefig(f'./backend/intrestingdata/predicted/{e[0]}.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()
            save(m.state_dict(), f'./backend/trained_models/{e[0]}.pt')

            with open('./backend/intrestingdata/results.json', 'a') as f:
                json.dump({int(time.time()) : { 'args' : args, 'mse' : mse_results, 'r2' : r2_results}}, f, indent=4)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Defined here
    params = {
        "b_size" : 8,
        "filter_size" : 3,
        "iterations" : 5,
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
            print("hello")
            main(**params)