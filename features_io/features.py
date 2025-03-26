import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs as sp
from mne_connectivity.viz import plot_connectivity_circle 
def coh(data):

    # Compute coherence
    events = mne.make_fixed_length_events(data, duration=8, overlap=0.0)
    epochs = mne.Epochs(data, events, tmin=0, tmax = 8, baseline=None, preload=True )
    con = sp(method = 'coh', data=epochs, faverage=False)
    print(epochs.info)
    plot_connectivity_circle(
        con.get_data(output='dense')[:, :, 0],  # Shape: (19, 19)
        node_names=epochs.info['ch_names'],
        title='wPLI in Alpha Band',
        facecolor='white',
        colormap='viridis',
        show=True)
    return(con)