def _sl(data_input: dict, freq: str, w1: int = 5, w2: int = 10):
    """
    Compute Synchronization Likelihood (SL) for EEG data across 19 channels, with downsampling to 19x100.

    Parameters:
    - data_input: dict, containing EEG data and sampling frequency
    - freq: str, frequency band ('delta', 'theta', 'alpha', 'beta', 'gamma')
    - w1: int, window for Theiler correction (default: 1 for 100 time points)
    - w2: int, window for time resolution sharpening (default: 2 for 100 time points)

    Returns:
    - sl_matrix: numpy array of shape (1, 19, 19), SL matrix
    """

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
    if data is None:
        raise ValueError("Failed to convert input to RawArray.")

    events = mne.make_fixed_length_events(data, duration=5, overlap=0.0)
    epochs = mne.Epochs(data, events, tmin=0, tmax=5, baseline=None, preload=True, verbose=False)

    if len(epochs) == 0:
        raise ValueError("No epochs created. Check input data or epoching parameters.")

    # Filter data in the specified frequency band
    epochs.filter(l_freq=fmin, h_freq=fmax, verbose=False)

    # Downsample epochs to 100 time points (assuming original sampling rate gives 15000 points over 5 seconds)
    original_sfreq = data.info['sfreq']
    target_sfreq = 25 / 5.0  # 20 Hz for 100 points over 5 seconds
    epochs.resample(sfreq=target_sfreq, verbose=False)

    # Initialize SL matrix
    num_channels = 19
    sl_matrix = np.zeros((num_channels, num_channels))

    # Compute SL for each epoch and average
    for epoch in epochs.get_data():  # Shape: (n_channels, n_time_points), e.g., (19, 100)
        if epoch.shape[0] != num_channels or epoch.shape[1] != 25:
            raise ValueError(f"Unexpected epoch shape: {epoch.shape}, expected ({num_channels}, 100)")

        time_points = epoch.shape[1]  # 100 time points
        temp_sl = np.zeros((num_channels, num_channels))

        # Validate window sizes
        if w1 >= time_points or w2 >= time_points or w1 < 0 or w2 < 0:
            raise ValueError(f"Invalid window sizes: w1={w1}, w2={w2} for {time_points} time points")

        # Compute SL between all channel pairs
        for k1 in range(num_channels):
            for k2 in range(k1, num_channels):  # Upper triangle to avoid redundant computation
                sync_sum = 0
                count = 0
                for i in range(time_points):
                    for j in range(time_points):
                        if abs(i - j) <= w1:
                            continue
                        if abs(i - j) <= w2 and i + w2 <= time_points and j + w2 <= time_points:
                            segment_i = epoch[k1, i:i+w2]
                            segment_j = epoch[k2, j:j+w2]
                            if len(segment_i) == w2 and len(segment_j) == w2:  # Ensure valid segments
                                sync = np.abs(np.corrcoef(segment_i, segment_j)[0, 1])
                                if not np.isnan(sync):  # Skip NaN correlations
                                    sync_sum += sync
                                    count += 1
                if count > 0:
                    temp_sl[k1, k2] = sync_sum / count
                    temp_sl[k2, k1] = temp_sl[k1, k2]  # Symmetrize

        # Normalize
        if np.max(temp_sl) != 0:
            temp_sl = np.clip(temp_sl / np.max(temp_sl), 0, 1)
        sl_matrix += temp_sl

    # Average across epochs
    sl_matrix /= len(epochs) if len(epochs) > 0 else 1

    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(sl_matrix, 0)

    # Reshape to (1, 19, 19) for compatibility with _stack_matrices
    return np.expand_dims(sl_matrix, axis=0)