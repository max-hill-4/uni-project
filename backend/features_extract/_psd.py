import numpy as np
from features_extract._utils import _epochtoRawArray
import mne
def _psd(data_input: dict, freq: str):


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
    data = _epochtoRawArray(data_input)
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

    # Transpose to match PyTorch channel-first format (1, n_channels, n_channels)
    psd_matrix = transpose(psd_matrix, (0, 2, 1))  # Shape: (1, n_channels, n_channels)
    print(psd_matrix.shape)
    return psd_matrix