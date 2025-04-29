import mne
from mne_connectivity import spectral_connectivity_epochs as sp
from numpy import transpose
from mne import create_info
from mne.io import RawArray
from mne.channels import make_standard_montage

class FeatureExtractor:
    def __init__(self, feature='coh'):
        self.feature = feature

    def get(self, data):
        if self.feature == 'coh':
            return self._coh(data)

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
    def _coh(data_input: dict, freq='alpha'):
        data = FeatureExtractor._epochtoRawArray(data_input)
        events = mne.make_fixed_length_events(data, duration=10, overlap=0.0)
        epochs = mne.Epochs(data, events, tmin=0, tmax=10, baseline=None, preload=True, verbose=False)
        con = sp(method='coh', data=epochs, fmin=4, fmax=8, faverage=True, verbose=False)
        data = con.get_data(output='dense')
        data = transpose(data, (2, 0, 1))  # PyTorch uses channel-first format
        return data
