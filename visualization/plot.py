
from mne.io import RawArray 
def raw_plot(data):
    spectrum = data.compute_psd()
    spectrum.plot(average=True, picks="data", exclude="bads", amplitude=False)
    data.plot_sensors(block=True)