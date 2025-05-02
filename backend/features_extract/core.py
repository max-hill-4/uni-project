from features_extract import _coh, _pdc, _lc, _psd, _sl


class FeatureExtractor:

    def __init__(self, feature_freq): 
        self.feature_freq = feature_freq
        print(feature_freq) 
   
    def get(self, data):
        
        matrices = []
        
        for pair in self.feature_freq:
            feature, freq = next(iter(pair.items()))
            if feature == 'coh':
                matrices.append(_coh(data, freq))
            elif feature == 'pdc':
                matrices.append(_pdc(data, freq))
            elif feature == 'lc':
                matrices.append(_lc(data, freq))
            elif feature == 'psd':
                matrices.append(_psd(data, freq))
            elif feature == 'sl':
                matrices.append(_sl(data, freq))
            else:
                raise ValueError(f"Unsupported feature: {feature == 'coh'}")
        
        return self._stack_matrices(matrices)