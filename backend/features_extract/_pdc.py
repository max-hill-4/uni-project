import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.signal import resample
def _pdc(data_input: dict, freq):
    order = 10


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
