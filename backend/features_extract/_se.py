from features_extract._utils import _epochtoRawArray
import mne
import numpy as np
import pywt

def _se(data_input: dict, freq: str):
    """
    Compute Shannon entropy for EEG data across 19 channels using ODWT relative energies,
    producing an upper triangular matrix.

    Parameters:
    - data_input: dict, containing EEG data and sampling frequency
    - freq: str, frequency band ('delta', 'theta', 'alpha', 'beta', 'gamma')

    Returns:
    - entropy_matrix: numpy array of shape (1, 19, 19), upper triangular Shannon entropy matrix
    """

    # Define frequency bands and wavelet levels (assuming sfreq=100 Hz)
    FREQ_BANDS = {
        'delta': (0.5, 4),  # Level 5: ~0.78–3.13 Hz
        'theta': (4, 8),    # Level 4: ~3.13–6.25 Hz
        'alpha': (8, 13),   # Level 3: ~6.25–12.5 Hz
        'beta': (13, 20),   # Level 2: ~12.5–25 Hz
        'gamma': (20, 50)   # Level 1: ~25–50 Hz
    }
    LEVEL_MAP = {'delta': 5, 'theta': 4, 'alpha': 3, 'beta': 2, 'gamma': 1}

    # Validate frequency band
    if freq not in FREQ_BANDS:
        raise ValueError(f"Frequency must be one of: {', '.join(FREQ_BANDS.keys())}.")
    fmin, fmax = FREQ_BANDS[freq]
    wavelet_level = LEVEL_MAP[freq]

    # Convert input to RawArray and create epochs
    data = _epochtoRawArray(data_input)
    if data is None:
        raise ValueError("Failed to convert input to RawArray.")

    events = mne.make_fixed_length_events(data, duration=5, overlap=0.0)
    epochs = mne.Epochs(data, events, tmin=0, tmax=5, baseline=None, preload=True, verbose=False)

    if len(epochs) == 0:
        raise ValueError("No epochs created. Check input data or epoching parameters.")

    # Filter data in the specified frequency band
    epochs.filter(l_freq=fmin, h_freq=fmax, verbose=False,method='iir', phase='zero')

    # Downsample to 100 Hz (500 points over 5 seconds)
    target_sfreq = 100.0
    epochs.resample(sfreq=target_sfreq, verbose=False)
    # Initialize entropy matrix
    num_channels = 19
    entropy_matrix = np.zeros((num_channels, num_channels))

    # Wavelet settings
    wavelet = 'sym5'  # Symlet wavelet as proxy for orthogonal cubic spline
    max_level = 5

    # Compute entropy for each epoch and average
    for epoch in epochs.get_data():  # Shape: (n_channels, n_time_points), e.g., (19, 500)
        if epoch.shape[0] != num_channels or epoch.shape[1] != 500:
            raise ValueError(f"Unexpected epoch shape: {epoch.shape}, expected ({num_channels}, 500)")

        temp_entropy = np.zeros((num_channels, num_channels))

        # Compute ODWT for each channel
        coeffs_all = []
        for ch in range(num_channels):
            signal = epoch[ch, :]
            coeffs = pywt.wavedec(signal, wavelet, level=max_level, mode='periodization')
            coeffs_all.append(coeffs)  # List of [cA5, cD5, cD4, cD3, cD2, cD1]

        # Compute Shannon entropy for channel pairs (upper triangle only)
        for k1 in range(num_channels):
            for k2 in range(k1 + 1, num_channels):  # Upper triangle, exclude diagonal
                # Get coefficients for the specified level
                c1 = coeffs_all[k1][max_level - wavelet_level + 1]  # cDj for level j
                c2 = coeffs_all[k2][max_level - wavelet_level + 1]

                # Total energy: sum of squared coefficients across all levels
                e_tot1 = sum(np.sum(np.abs(c)**2) for c in coeffs_all[k1])
                e_tot2 = sum(np.sum(np.abs(c)**2) for c in coeffs_all[k2])

                # Energy at level j
                e_j1 = np.sum(np.abs(c1)**2)
                e_j2 = np.sum(np.abs(c2)**2)

                # Relative energies
                p_j1 = e_j1 / e_tot1 if e_tot1 > 0 else 0
                p_j2 = e_j2 / e_tot2 if e_tot2 > 0 else 0

                # Shannon entropy: H = -sum(p * log2(p)) for the pair
                if p_j1 > 0 and p_j2 > 0:
                    # Average entropy of the two probabilities
                    entropy = -0.5 * (p_j1 * np.log2(p_j1) + p_j2 * np.log2(p_j2))
                    # Normalize by log2(2) = 1 to range [0, 1]
                    entropy_val = np.clip(entropy / np.log2(2), 0, 1)
                else:
                    entropy_val = 0

                temp_entropy[k1, k2] = entropy_val  # Only upper triangle

        entropy_matrix += temp_entropy

    # Average across epochs
    entropy_matrix /= len(epochs) if len(epochs) > 0 else 1

    # Ensure diagonal is 0
    np.fill_diagonal(entropy_matrix, 0)

    # Reshape to (1, 19, 19)
    return np.expand_dims(entropy_matrix, axis=0)