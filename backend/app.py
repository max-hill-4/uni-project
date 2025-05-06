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
import numpy as np
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

        print(tr_parps)
        train_data = DataLoader(train_dataset, batch_size=args["b_size"], shuffle=True, num_workers=4, collate_fn=data_io.dataloader.collate_fn)    
        test_data = DataLoader(test_dataset, batch_size=args["b_size"], shuffle=True, num_workers=4, collate_fn=data_io.dataloader.collate_fn)
        m = analysis.models.EEGCNN(filter_size=args["filter_size"], num_classes=len(args['hormones']), in_channels=args["in_channels"])
        a = analysis.train.model(m, train_data, test_data, iterations=args[ "iterations" ])

        a.train()

        predictions, truth = a.predict()

        predicted_classes = torch.argmax(predictions, dim=1)

        correct = (predicted_classes == truth).sum().item()
        accuracy = correct / truth.size(0)
        print(accuracy)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Defined here
    import os 
    print(os.getcwd())
    params = {
        "b_size" : 4,
        "filter_size" : 3,
        "iterations" : 6,
        "k_folds" : 10,
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