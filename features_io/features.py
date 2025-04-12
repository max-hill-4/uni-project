import mne
from mne_connectivity import spectral_connectivity_epochs as sp
from mne_connectivity.viz import plot_connectivity_circle 
from numpy import ndarray, transpose
from data_io import raw
# This wull for sure cook later, as will enable a design pattern to select different / multiple features 
class Feature():
    def __init__(self):
        pass

class Coherance(Feature):
    def coh(data_input: dict) -> ndarray:
        # Compute coherence
        
        data = raw.epochtoRawArray(data_input)
        events = mne.make_fixed_length_events(data, duration=10, overlap=0.0)
        epochs = mne.Epochs(data, events, tmin=0, tmax = 10, baseline=None, preload=True )
        con = sp(method = 'coh', data=epochs, fmin=8, fmax=12, faverage=True) # This is only for the alpha band !  


        # this is going to return a single 19x19 matrix of a single channel.  
        data = con.get_data(output='dense')
        data = transpose(data, (2, 0, 1)) # pytorch uses channel first tensor.
        print(f"shape of feature matrix is {data.shape}")
        return data
    def plot():
        pass # this wont work at the momeny becuase the class needs to be instatiated to use con.
        '''
        plot_connectivity_circle(
            con.get_data(output='dense')[:, :, 0],  # Shape: (19, 19)
            node_names= epochs.info['ch_names'],
            title='wPLI in Alpha Band',
            facecolor='white',
            colormap='viridis',
            show=True)
        '''