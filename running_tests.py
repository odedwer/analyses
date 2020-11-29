from preprocess_utilities import *
import sys
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
components = range(20) # number of components to show

if __name__ == "__main__":
    mpl.use("TkAgg")
    raw = mne.io.read_raw_fif("hpf1-raw.fif")
    raw.load_data()
    raw.set_montage(montage=mne.channels.make_standard_montage('biosemi256', head_size=0.089), raise_if_subset=False)
    #print("plotting full psd...")
    #raw.plot_psd(fmin=0,fmax=300,picks=range(20),n_fft=10*2048)
    #plt.show()
    # print("plotting short psd...")
    # raw.plot_psd(fmin=0,fmax=30,picks=range(20),n_fft=10*2048)
    # plt.show()
    ica = mne.preprocessing.read_ica("hpf1-ica.fif")
    #print("plotting components...")
    #ica.plot_components(outlines='skirt', picks=components, show=False)
    #plt.show()

    #print("plotting properties...")
    # the beginning of each components group to be shown
    comp_jumps = np.linspace(0, ica.n_components_, int(ica.n_components_ / 8) + 1)
    for i in range(len(comp_jumps)): # go over the components and show 8 each time
        if input("stop plotting? (Y/N)") == "Y":
            break
        comps = range(int(comp_jumps[i]), int(comp_jumps[i + 1]))
        print("plotting from component "+str(comps))
     #   ica.plot_properties(raw, picks=comps, show=False)  # plot component properties
        plot_correlations(ica, raw, components=comps,
                          picks=['A1','Nose','RHEOG','LHEOG','RVEOGS','RVEOGI','M1','M2','LVEOGI'])
        ica.plot_sources(raw, picks=comps, show=False) #plot sources
        plt.show()