import data_io
import visualization
import os
import features_io
import numpy as np

files = (os.listdir(r'C:\Users\Max\Downloads\N2'))
testEpoch = data_io.raw.epochtoRawArray(r'C:\Users\Max\Downloads\N2\bdc14_Z4_0253.mat')
coh = features_io.features.coh(testEpoch)
#visualization.plot.raw_plot(testEpoch)
#visualization.plot.raw_sensors(testEpoch)