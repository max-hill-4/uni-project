import mne
import scipy.io

def epochtoRawArray(filepath) :
    mat_data = scipy.io.loadmat(filepath)
    info = mne.create_info(
        ch_names=[f'EEG Channel {i+1}' for i in range(19)],
        sfreq=500, # Need to check with Christos ! 
        ch_types='eeg'
    )
    return mne.io.RawArray(mat_data['current_epoch'], info)