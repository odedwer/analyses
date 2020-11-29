## Gal Chen
## use this script to load a file and print a specific graph...playground.


# %
# ##import
import numpy as np
from preprocess_utilities import *
import matplotlib as mpl

if __name__ == "__main__":
    mpl.use("TkAgg")
    root = Tk()
    root.withdraw()
    # upload raw files AFTER robust detrending
    raws = read_bdf_files(preload=True)
    # #detrended_raws = load_raws_from_mat('detrended_ord10_10s_window.mat', raws)
    # concatenate to one raw file
    raw = mne.concatenate_raws(raws)
    copy_raw = raw.copy() #make a copy before adding the new channel
    raw.filter(l_freq=1, h_freq=None)
    raw.plot(duration=6, n_channels=10)
    plt.show()