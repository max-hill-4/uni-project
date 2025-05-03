import numpy as np
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