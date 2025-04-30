from mne_connectivity import spectral_connectivity_epochs as sp
from numpy import transpose, ceil, zeros, triu_indices,tril_indices
from mne import create_info
from mne.io import RawArray
from mne.channels import make_standard_montage
import numpy as np
import mne


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
            elif feature == 'pdc':
                matrices.append(self._pdc(data, freq))
            else:
                raise ValueError(f"Unsupported feature: {feature == 'coh'}")
        
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
        events = mne.make_fixed_length_events(data, duration=5, overlap=0.0)
        epochs = mne.Epochs(data, events, tmin=0, tmax=5, baseline=None, preload=True, verbose=False)
        
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
        print('coh calcd')
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
    def _pdc(data_input: dict, freq):
        order = 10
        n_fft=512

        import numpy as np
        from statsmodels.tsa.vector_ar.var_model import VAR
        from scipy.signal import resample
        data = data_input['current_epoch']
        sampling_rate = 500
        freq_band = [8, 12]
        n_samples = 200
        n_channels, _ = data.shape
        
        # Downsample data
        data_downsampled = resample(data, n_samples, axis=1)
        new_sampling_rate = sampling_rate * n_samples / data.shape[1]
        
        # Fit minimal MVAR model
        model = VAR(data_downsampled.T)
        results = model.fit(order)
        ar_coefs = results.coefs  # Shape: (order, n_channels, n_channels)
        
        # Single frequency (midpoint of band)
        f_min, f_max = freq_band
        f = (f_min + f_max) / 2 / new_sampling_rate  # Normalized frequency
        
        # Fourier transform of AR coefficients
        A_f = np.eye(n_channels, dtype=np.complex64)
        for lag in range(order):
            A_f -= ar_coefs[lag] * np.exp(-2j * np.pi * f * (lag + 1))
        
        # Vectorized PDC
        abs_A_f = np.abs(A_f)
        denominators = np.sqrt(np.sum(abs_A_f**2, axis=0))
        pdc = abs_A_f / denominators[np.newaxis, :]
        np.fill_diagonal(pdc, 0)
        
        # Output shape (1, n_channels, n_channels)
        return pdc[np.newaxis, :, :].astype(np.float32)

    @staticmethod
    def compute_lagged_correlation(data, lag=1, n_samples=1000):
        """
        Compute lagged correlation as a fast proxy for directed connectivity.
        
        Parameters:
        - data: numpy array, shape (n_channels, n_samples)
        - lag: time lag for correlation (in samples, e.g., 1)
        - n_samples: number of samples after downsampling (e.g., 1000)
        
        Returns:
        - corr: numpy array, shape (1, n_channels, n_channels), lagged correlation matrix
        """
        n_channels, _ = data.shape
        
        # Downsample data
        data_downsampled = resample(data, n_samples, axis=1)
        
        # Normalize data
        data_norm = (data_downsampled - np.mean(data_downsampled, axis=1, keepdims=True)) / np.std(data_downsampled, axis=1, keepdims=True)
        
        # Compute lagged correlation
        corr = np.zeros((n_channels, n_channels), dtype=np.float32)
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    # Correlation between data[i, t] and data[j, t-lag]
                    corr[i, j] = np.corrcoef(data_norm[i, lag:], data_norm[j, :-lag])[0, 1]
        
        return corr[np.newaxis, :, :]