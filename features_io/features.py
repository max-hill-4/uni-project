import mne
from mne_connectivity import spectral_connectivity_epochs as sp
from numpy import transpose, ceil, zeros, triu_indices,tril_indices
from mne import create_info
from mne.io import RawArray
from mne.channels import make_standard_montage


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
        events = mne.make_fixed_length_events(data, duration=10, overlap=0.0)
        epochs = mne.Epochs(data, events, tmin=0, tmax=10, baseline=None, preload=True, verbose=False)
        
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
        con = sp(method='coh', data=epochs, fmin=fmin, fmax=fmax, faverage=True, verbose=False)
        
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
        
            print("UPPER MATRIX UPPER UPPER ! : ", output[i][upper_tri])
            print("LOWER MATRIX LOWER LOWER! : ", output[i][lower_tri])
        # Process last unpaired matrix (if odd count)
        if n_singles > 0:
            output[-1] = matrices[-1][0]  # Full matrix in last channel
        print(output.shape)
        return output  # Shape: [total_channels, 19, 19]