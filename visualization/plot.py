import numpy as np
from mne.io import RawArray 
def raw_plot(data):
    data.plot(block=True, scalings='auto')
def raw_sensors(data):
    data.plot_sensors(block=True, show_names=True)