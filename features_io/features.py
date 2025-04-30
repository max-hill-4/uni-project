import mne
from mne_connectivity import spectral_connectivity_epochs as sp
from numpy import transpose, ceil, zeros, triu_indices,tril_indices
from mne import create_info
from mne.io import RawArray
from mne.channels import make_standard_montage
import numpy as np
import numpy as np
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs
from mne.datasets import sample

class FeatureExtractor:
    def __init__(self, feature_freq): #could change to dict ? kley:value pariut tehcincally!
        self.feature_freq = feature_freq
        print(feature_freq) 
   
    def get(self, data):
        
        matrices = []
        
        for pair in self.feature_freq:
            feature, freq = next(iter(pair.items()))
            if feature == 'coh':
                matrices.append(self._coh(data, freq))
            else:
                raise ValueError(f"Unsupported feature: {feature}")
        
        return self._stack_matrices(matrices)
        
    

    @staticmethod
    def _epochtoRawArray(data: dict):
        info = create_info(
            ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz',
                      'C3', 'C4', 'Cz', 'T3', 'T4', 'T5', 'T6',
                      'P3', 'P4', 'Pz', 'O1', 'O2'],
            sfreq=500,
            ch_types='eeg'
        )
        raw = RawArray(data['current_epoch'], info, verbose=False)
        montage = make_standard_montage('standard_1020')
        raw.set_montage(montage)
        return raw

    @staticmethod
    def _coh(data_input: dict, freq):
        data = FeatureExtractor._epochtoRawArray(data_input)
        events = mne.make_fixed_length_events(data, duration=2, overlap=0.0)
        epochs = mne.Epochs(data, events, tmin=0, tmax=2, baseline=None, preload=True, verbose=False)
        
                # Define frequency bands
        FREQ_BANDS = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        # Compute coherence based on frequency band
        if freq not in FREQ_BANDS:
            raise ValueError(f"Frequency must be one of: {', '.join(FREQ_BANDS.keys())}.")
        fmin, fmax = FREQ_BANDS[freq]
        con = sp(method='pli', data=epochs, fmin=fmin, fmax=fmax, faverage=True, verbose=False)
        
        data = con.get_data(output='dense')
        data = transpose(data, (2, 0, 1))  # PyTorch uses channel-first format
        return data

    @staticmethod
    def _stack_matrices(matrices):
        if not matrices:
            raise ValueError("No matrices to stack.")
        
        n_matrices = len(matrices)
        max_channels = 5  # Adjust as needed
        
        # Validate shapes
        for mat in matrices:
            if mat.shape != (1, 19, 19):
                raise ValueError(f"All matrices must be [1, 19, 19]. Got {mat.shape}")
        
        # Calculate output channels (pairs + remaining singles)
        n_pairs = n_matrices // 2
        n_singles = n_matrices % 2
        total_channels = n_pairs + n_singles
        
        if total_channels > max_channels:
            raise ValueError(f"Max {max_channels} channels allowed. Got {total_channels}.")
        
        output = zeros((total_channels, 19, 19))
        lower_tri = tril_indices(19, k=-1)
        upper_tri = triu_indices(19, k=1)
        
        # Process pairs
        for i in range(n_pairs):
            lower_mat = matrices[2*i][0]    # [19, 19]
            upper_mat = matrices[2*i+1][0]  # [19, 19]
            
            # Combine into one channel
            output[i][lower_tri] = lower_mat[lower_tri]
            output[i][upper_tri] = upper_mat[lower_tri]
        
        # Process last unpaired matrix (if odd count)
        if n_singles > 0:
            output[-1] = matrices[-1][0]  # Full matrix in last channel
        return output  # Shape: [total_channels, 19, 19]
    
    @staticmethod
    def _sloreta(data_input: dict, freq: str):
        """Compute sLORETA source estimates for a given frequency band.
        
        Parameters:
        -----------
        data_input : dict 
            Input data dictionary (matches your PLI example format)
        freq : str
            Frequency band name ('delta', 'theta', etc.)
            
        Returns:
        --------
        stc_data : ndarray
            Source time courses in shape (n_epochs, n_vertices)
        """
        # Convert input to MNE Raw (same as your PLI method)
        data = FeatureExtractor._epochtoRawArray(data_input)
        events = mne.make_fixed_length_events(data, duration=2, overlap=0.0)
        epochs = mne.Epochs(data, events, tmin=0, tmax=2, baseline=None, preload=True, verbose=False)
        
        # Define frequency bands (identical to your PLI implementation)
        FREQ_BANDS = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        if freq not in FREQ_BANDS:
            raise ValueError(f"Frequency must be one of: {', '.join(FREQ_BANDS.keys())}.")
        fmin, fmax = FREQ_BANDS[freq]
        
        # Band-pass filter the epochs (critical for sLORETA)
        epochs_filtered = epochs.copy().filter(fmin, fmax, verbose=False)
        
        # --- sLORETA-specific steps ---
        # 1. Load forward solution (replace with your own)
        # This is a placeholder - you need your subject's head model!
        fwd = mne.read_forward_solution('your_forward_model-fwd.fif')  
        
        # 2. Compute noise covariance from baseline
        noise_cov = mne.compute_covariance(epochs_filtered, tmin=0, tmax=None, verbose=False)
        
        # 3. Create inverse operator
        inv = mne.minimum_norm.make_inverse_operator(
            epochs_filtered.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False
        )
        
        # 4. Apply sLORETA to each epoch
        stcs = mne.minimum_norm.apply_inverse_epochs(
            epochs_filtered, inv, lambda2=1./9., 
            method="sLORETA", pick_ori=None, verbose=False
        )
        
        # Convert to numpy array and average over time
        stc_data = np.array([stc.data.mean(axis=1) for stc in stcs])  # (n_epochs, n_vertices)
        
        return stc_data