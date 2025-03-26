import mne
from mne_connectivity import spectral_connectivity_epochs as sp
from mne_connectivity.viz import plot_connectivity_circle 
from numpy import ndarray


# This wull for sure cook later, as will enable a design pattern to select different / multiple features 
class Feature():
    def __init__(self):
        pass

class Coherance(Feature):
    def coh(data) -> ndarray:
        # Compute coherence
        events = mne.make_fixed_length_events(data, duration=10, overlap=0.0)
        epochs = mne.Epochs(data, events, tmin=0, tmax = 10, baseline=None, preload=True )
        con = sp(method = 'coh', data=epochs, faverage=False)
        '''
        plot_connectivity_circle(
            con.get_data(output='dense')[:, :, 0],  # Shape: (19, 19)
            node_names=epochs.info['ch_names'],
            title='wPLI in Alpha Band',
            facecolor='white',
            colormap='viridis',
            show=True)
        '''
        # returns square matrix 
        return(con.get_data())
