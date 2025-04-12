from mne.io import RawArray
from mne import create_info
from mne.channels import make_standard_montage
from scipy.io import loadmat
import numpy as np
def epochtoRawArray(data:dict) :
    info = create_info(
        ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz',
            'C3', 'C4', 'Cz',
            'T3', 'T4', 'T5', 'T6',
            'P3', 'P4', 'Pz',
            'O1', 'O2'],
        sfreq=500,
        ch_types='eeg'
    )
    raw = RawArray(data['current_epoch'], info, verbose=False)
    montage = make_standard_montage('standard_1020') # Or another montage
    raw.set_montage(montage)
    return raw
