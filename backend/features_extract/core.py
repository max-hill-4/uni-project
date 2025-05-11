from features_extract._coh import _coh
from features_extract._psd import _psd
from features_extract._lc import _lc
from features_extract._pdc import _pdc
from features_extract._se import _se


from features_extract._utils import _stack_matrices

class FeatureExtractor:

    def __init__(self, feature_freq): 
        self.feature_freq = feature_freq
        print(feature_freq) 
   
    def get(self, data):
        
        matrices = []
        
        for pair in self.feature_freq:
            print(f"creating first matrix of {pair}")
            feature, freq = next(iter(pair.items()))
            if feature == 'coh':
                matrices.append(_coh(data, freq))
            elif feature == 'pdc':
                matrices.append(_pdc(data, freq))
            elif feature == 'lc':
                matrices.append(_lc(data, freq))
            elif feature == 'psd':
                matrices.append(_psd(data, freq))
            elif feature == 'se':
                matrices.append(_se(data, freq))
            else:
                raise ValueError(f"Unsupported feature: {feature == 'coh'}")
        
        return _stack_matrices(matrices)