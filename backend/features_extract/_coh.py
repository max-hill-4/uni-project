from features_extract import _utils
from mne import make_fixed_length_events, Epochs
from mne_connectivity import spectral_connectivity_epochs as sp

def _coh(data_input: dict, freq):
    data = _utils._epochtoRawArray(data_input)
    events = make_fixed_length_events(data, duration=5, overlap=0.0)
    epochs = Epochs(data, events, tmin=0, tmax=5, baseline=None, preload=True, verbose=False)
    
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
    
    # Apply RobustScaler
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    original_shape = data.shape
    data = scaler.fit_transform(data.reshape(-1, original_shape[-1])).reshape(original_shape)
    
    data = transpose(data, (2, 0, 1))  # PyTorch uses channel-first format
    return data
