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
    fig = plt.figure(figsize=(8, 6))
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
    
    return fig, saliency


import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap
import networkx as nx
import numpy as np
from scipy.linalg import eigh
def compute_topo_map(saliency):
    """
    Compute saliency map and plot it as a topomap using MNE.
    
    Args:
        model: PyTorch/TensorFlow model.
        input_tensor: Input EEG data (shape: [n_channels, n_times] or [n_channels, n_channels] for connectivity).
        ch_names: List of channel names (e.g., ['Fp1', 'Fp2', ...]).
        montage: MNE montage name (default: 'standard_1020').
    """
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz',
                'C3', 'C4', 'Cz', 'T3', 'T4', 'T5', 'T6',
                'P3', 'P4', 'Pz', 'O1', 'O2'] 

    
    # Create MNE info object
    info = mne.create_info(ch_names, sfreq=500., ch_types='eeg')  # Adjust sfreq as needed
    info.set_montage('standard_1020')

    G = nx.from_numpy_array(saliency, create_using=nx.DiGraph)

    pagerank_scores = nx.pagerank(G) 
    s = np.array(list(pagerank_scores.values()))
    # Plot topomap
    fig, ax = plt.subplots(figsize=(8, 6))
    im, _ = plot_topomap(
        s,
        info, 
        cmap='jet', 
        sensors=True, 
        contours=6, 
        show=False,
        axes=ax,
        names = ch_names
    )
    fig.colorbar(im, ax=ax, shrink=0.6, label='Saliency (|∇output|)')
    
    plt.tight_layout()
    ax.set_title('EEG Saliency Map (10-20 System)')
    return fig