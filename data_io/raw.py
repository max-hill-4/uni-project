from mne.io import RawArray
from mne import create_info
from mne.channels import make_standard_montage
from scipy.io import loadmat

def epochtoRawArray(filepath) :
    mat_data = loadmat(filepath)
    info = create_info(
        ch_names=[f'EEG Channel {i+1}' for i in range(19)],
        sfreq=500, # Need to check with Christos ! 
        ch_types='eeg'
    )
    raw = RawArray(mat_data['current_epoch'], info)
    
    return raw