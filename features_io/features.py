from mne_connectivity import spectral_connectivity_epochs as sp
from numpy import transpose, ceil, zeros, triu_indices,tril_indices
from mne import create_info
from mne.io import RawArray
from mne.channels import make_standard_montage
import numpy as np
import mne
from mne.time_frequency import psd_array_welch
from numpy import transpose

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
            elif feature == 'lc':
                matrices.append(self._pdc(data, freq))
            elif feature == 'psd':
                matrices.append(self._sl(data, freq))
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
        
        # Process last unpaired matrix (if odd count)
        if n_singles > 0:
            output[-1] = matrices[-1][0]  # Full matrix in last channel
        return output  # Shape: [total_channels, 19, 19]
    
    @staticmethod
    def _pdc(data_input: dict, freq):
        order = 10

        import numpy as np
        from statsmodels.tsa.vector_ar.var_model import VAR
        from scipy.signal import resample
        data = data_input['current_epoch']
        sampling_rate = 500

        n_samples = 200 
        n_channels, _ = data.shape
         
        # Downsample data
        data_downsampled = resample(data, n_samples, axis=1)
        new_sampling_rate = sampling_rate * n_samples / data.shape[1]
        
        # Fit minimal MVAR model
        model = VAR(data_downsampled.T)
        results = model.fit(order)
        ar_coefs = results.coefs  # Shape: (order, n_channels, n_channels)

        FREQ_BANDS = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
 
        # Single frequency (midpoint of band)
        f_min, f_max = FREQ_BANDS[freq]
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
    def _lc(data_input, lag=1, n_samples=1000):
        """
        Compute lagged correlation as a fast proxy for directed connectivity.
        
        Parameters:
        - data: numpy array, shape (n_channels, n_samples)
        - lag: time lag for correlation (in samples, e.g., 1)
        - n_samples: number of samples after downsampling (e.g., 1000)
        
        Returns:
        - corr: numpy array, shape (1, n_channels, n_channels), lagged correlation matrix
        """
        data = data_input['current_epoch']
        n_channels, _ = data.shape
        
        from scipy.signal import resample
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

    @staticmethod
    def _psd(data_input: dict, freq: str, method='diagonal'):


        """
        Compute a square matrix based on PSD for the specified frequency band.
        
        Parameters:
        - data_input (dict): Input data for epochs.
        - freq (str): Frequency band (e.g., 'delta', 'theta').
        - method (str): 'diagonal' for per-channel PSD matrix, 'connectivity' for PSD-based connectivity.
        
        Returns:
        - np.ndarray: 3D array with shape (1, n_channels, n_channels), e.g., (1, 19, 19).
        """
        # Convert input to MNE RawArray
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

        if freq not in FREQ_BANDS:
            raise ValueError(f"Frequency must be one of: {', '.join(FREQ_BANDS.keys())}.")
        fmin, fmax = FREQ_BANDS[freq]

        if method == 'diagonal':
            # Compute PSD using Welch's method
            psd, freqs = epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax, n_fft=256, n_jobs=1).get_data(return_freqs=True)
            # Average PSD across epochs and frequencies
            psd_mean = psd.mean(axis=0).mean(axis=1)  # Shape: (n_channels,)
            
            # Create a diagonal square matrix
            n_channels = len(psd_mean)
            psd_matrix = np.zeros((n_channels, n_channels))
            np.fill_diagonal(psd_matrix, psd_mean)
            
            # Add leading dimension to match (1, n_channels, n_channels)
            psd_matrix = np.expand_dims(psd_matrix, axis=0)  # Shape: (1, n_channels, n_channels)

        elif method == 'connectivity':
            # Compute connectivity (e.g., coherence) based on PSD
            con = sp(
                epochs, method='coh', fmin=fmin, fmax=fmax, faverage=True, verbose=False
            )
            psd_matrix = con.get_data(output='dense')  # Shape: (n_channels, n_channels, 1)
            # Ensure 3D shape (1, n_channels, n_channels)
            if psd_matrix.ndim == 2:
                psd_matrix = np.expand_dims(psd_matrix, axis=0)  # Add leading dimension
        
        else:
            raise ValueError("Method must be 'diagonal' or 'connectivity'.")

        # Transpose to match PyTorch channel-first format (1, n_channels, n_channels)
        psd_matrix = transpose(psd_matrix, (0, 2, 1))  # Shape: (1, n_channels, n_channels)
        print(psd_matrix.shape)
        return psd_matrix
    @staticmethod
    def _sl(data_input: dict, freq: str, w1: int = 10, w2: int = 20):
        """
        Compute Synchronization Likelihood (SL) for EEG data across 19 channels.

        Parameters:
        - data_input: dict, containing EEG data and sampling frequency
        - freq: str, frequency band ('delta', 'theta', 'alpha', 'beta', 'gamma')
        - w1: int, window for Theiler correction
        - w2: int, window for time resolution sharpening

        Returns:
        - sl_matrix: numpy array of shape (19, 19), SL matrix
        - metrics: dict, graph metrics
        """

        import numpy as np
        import mne
        import networkx as nx
        from scipy.signal import welch

        # Define frequency bands
        FREQ_BANDS = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        # Validate frequency band
        if freq not in FREQ_BANDS:
            raise ValueError(f"Frequency must be one of: {', '.join(FREQ_BANDS.keys())}.")
        fmin, fmax = FREQ_BANDS[freq]

        # Convert input to RawArray and create epochs
        data = FeatureExtractor._epochtoRawArray(data_input)
        events = mne.make_fixed_length_events(data, duration=5, overlap=0.0)
        epochs = mne.Epochs(data, events, tmin=0, tmax=5, baseline=None, preload=True, verbose=False)

        # Filter data in the specified frequency band
        epochs.filter(l_freq=fmin, h_freq=fmax, verbose=False)

        # Initialize SL matrix
        num_channels = 19
        sl_matrix = np.zeros((num_channels, num_channels))

        # Compute SL for each epoch and average
        for epoch in epochs.get_data():  # Shape: (1, channels, time_points)
            epoch = epoch.squeeze(0)  # Shape: (channels, time_points)
            time_points = epoch.shape[1]
            temp_sl = np.zeros((num_channels, num_channels))

            for k in range(num_channels):
                for i in range(time_points):
                    sync_sum = 0
                    count = 0
                    for j in range(time_points):
                        if abs(i - j) <= w1:
                            continue
                        if abs(i - j) <= w2 and i + w2 <= time_points and j + w2 <= time_points:
                            # Simplified synchronization measure (correlation in frequency band)
                            segment_i = epoch[k, i:i+w2]
                            segment_j = epoch[k, j:j+w2]
                            sync = np.abs(np.corrcoef(segment_i, segment_j)[0, 1])
                            sync_sum += sync
                            count += 1
                    temp_sl[k, k] += sync_sum / count if count > 0 else 0

            # Symmetrize and normalize
            temp_sl = (temp_sl + temp_sl.T) / 2
            temp_sl = np.clip(temp_sl / np.max(temp_sl) if np.max(temp_sl) != 0 else temp_sl, 0, 1)
            sl_matrix += temp_sl

        # Average across epochs
        sl_matrix /= len(epochs)

        # Set diagonal to 0 (no self-loops)
        np.fill_diagonal(sl_matrix, 0)

        # Compute graph metrics
        G = nx.from_numpy_array(sl_matrix)
        metrics = {}

        # Small-world property
        clustering_coeff = nx.average_clustering(G, weight='weight')
        char_path_length = nx.average_shortest_path_length(G, weight='weight') if nx.is_connected(G) else float('inf')
        metrics['small_world_sigma'] = clustering_coeff / char_path_length if char_path_length != 0 else 0

        # Clustering coefficient
        metrics['clustering_coefficient'] = clustering_coeff

        # Graph density
        metrics['graph_density'] = nx.density(G)

        # Characteristic path length
        metrics['characteristic_path_length'] = char_path_length

        # Efficiency
        metrics['efficiency'] = 1 / char_path_length if char_path_length != 0 else 0

        # Mean betweenness centrality
        betweenness = nx.betweenness_centrality(G, weight='weight')
        metrics['mean_betweenness_centrality'] = np.mean(list(betweenness.values()))

        return sl_matrix