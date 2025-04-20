import mne
from mne_connectivity import spectral_connectivity_epochs as sp
from numpy import ndarray, transpose
from data_io import raw
import numpy as np
from scipy.io import loadmat

# This wull for sure cook later, as will enable a design pattern to select different / multiple features 
class Feature():
    def __init__(self):
        pass

class Coherance(Feature):
    def coh(data_input: dict,freq='alpha') -> tensor: 
        
        data = raw.epochtoRawArray(data_input)
        events = mne.make_fixed_length_events(data, duration=10, overlap=0.0)
        epochs = mne.Epochs(data, events, tmin=0, tmax = 10, baseline=None, preload=True, verbose=False)
        con = sp(method = 'coh', data=epochs, fmin=4, fmax=8, faverage=True, verbose=False) # This is only for the alpha band !  

        data = con.get_data(output='dense')
        data = transpose(data, (2, 0, 1)) # pytorch uses channel first tensor.
        return data

