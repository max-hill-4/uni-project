import data_io
import visualization
import os
import features_io
import numpy as np

testEpoch = data_io.raw.epochtoRawArray(r'./raw_data/bdc14_A1_0026.mat')
coh = features_io.features.Coherance.coh(testEpoch)

# how bad would it be to pilot the main program with a numpy array, and when using 
# feature selection or visualsation
#visualization.plot.raw_plot(testEpoch)
#visualization.plot.raw_sensors(testEpoch)