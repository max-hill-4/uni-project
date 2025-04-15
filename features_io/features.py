import mne
from mne_connectivity import spectral_connectivity_epochs as sp
from mne_connectivity.viz import plot_connectivity_circle 
from numpy import ndarray, transpose
from data_io import raw
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
import pywt

# This wull for sure cook later, as will enable a design pattern to select different / multiple features 
class Feature():
    def __init__(self):
        pass

class Coherance(Feature):
    def coh(data_input: dict, ) -> ndarray:
        # Compute coherence
        
        data = raw.epochtoRawArray(data_input)
        events = mne.make_fixed_length_events(data, duration=10, overlap=0.0)
        epochs = mne.Epochs(data, events, tmin=0, tmax = 10, baseline=None, preload=True, verbose=False)
        con = sp(method = 'coh', data=epochs, fmin=30, fmax=100, faverage=True, verbose=False) # This is only for the alpha band !  

        # this is going to return a single 19x19 matrix of a single channel.  
        data = con.get_data(output='dense')
        data = transpose(data, (2, 0, 1)) # pytorch uses channel first tensor.
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
class RWE(Feature): 
    import numpy as np

def relative_wavelet_entropy(signal, wavelet='db4', level=4):
    """
    Compute the Relative Wavelet Entropy (RWE) of a signal.

    Parameters:
    - signal: The input 1D signal (array-like).
    - wavelet: Wavelet name (default: 'db4').
    - level: Decomposition level (default: 4).

    Returns:
    - RWE: The Relative Wavelet Entropy of the signal.
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Calculate the energy of each coefficient
    energies = np.array([np.sum(c**2) for c in coeffs])
    
    # Normalize to get probabilities
    probabilities = energies / np.sum(energies)
    
    # Compute Shannon entropy of probabilities
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))  # Adding a small term to avoid log(0)
    
    # Compute maximum possible entropy
    max_entropy = np.log2(len(probabilities))
    
    # Calculate Relative Wavelet Entropy
    RWE = shannon_entropy / max_entropy

    return RWE
# Example Usage
signal = np.sin(np.linspace(0, 2 * np.pi, 100))  # Example: a sine wave signal


if __name__ == '__main__':
    c1 = loadmat(r'/home/student/Documents/uni-project/raw_data/bdc14_C4_0090.mat')['current_epoch'][1]
    c2 = loadmat(r'/home/student/Documents/uni-project/raw_data/bdc14_B4_0087.mat')['current_epoch'][1]
    m, l, w1, w2, pRef = 2, 5, 10, 50, 10
    rwe_value = relative_wavelet_entropy(c1)
    print("Relative Wavelet Entropy:", rwe_value)
    # Compute SL
    sl_value = SyncLikelyhood.synchronization_likelihood(c1, c2, m, l, w1, w2, pRef)
    print("Synchronization Likelihood (SL):", sl_value)