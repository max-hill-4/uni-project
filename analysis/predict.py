import torch
class predict():
    def __init__(m):
        m.eval()        
        with torch.no_grad():  # Disable gradient calculations for inference
            for batch in dataloader:
                data, labels = batch['data'].to(device), batch['label'].to(device)  # Move data and labels to the device
                
                # Get predictions from the model
                predictions = m(data)
                
                # Store predictions and labels
                all_predictions.append(predictions.cpu())  # Store predictions on the CPU for evaluation later
                all_labels.append(labels.cpu())  # Store labels on the CPU

        # After collecting predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)  # Ensure proper concatenation along the batch dimension
        all_labels = torch.cat(all_labels, dim=0)