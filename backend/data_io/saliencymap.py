import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def compute_saliency_map(model, input_tensor):
    # Channel names for EEG electrodes
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz',
                'C3', 'C4', 'Cz', 'T3', 'T4', 'T5', 'T6',
                'P3', 'P4', 'Pz', 'O1', 'O2']
    
    # Set model to evaluation mode
    model.eval()
    
    # Ensure input tensor requires gradients
    input_tensor = input_tensor.to('cpu').requires_grad_(True)
    
    # Forward pass
    output = model(input_tensor)  # Shape: [1, 1]
    
    # Backward pass to compute gradients
    output.sum().backward()
    
    # Compute saliency map (absolute value of gradients)
    saliency = input_tensor.grad.abs().squeeze().cpu().numpy()  # Shape: [19, 19]
    
    # Plot saliency map as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(saliency, xticklabels=ch_names, yticklabels=ch_names, 
                cmap='jet', cbar=True)
    plt.title('Saliency Map for EEG Coherence')
    plt.xlabel('EEG Channels')
    plt.ylabel('EEG Channels')
    
    # Rotate x-tick labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('saliency_map.png')
    plt.close()
    
    return saliency


