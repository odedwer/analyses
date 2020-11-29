from preprocess_utilities import *
import sys
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename

components = range(20)  # number of components to show

if __name__ == "__main__":
    mpl.use("tkAgg")
    root = Tk()
    root.withdraw()
    raw = mne.io.read_raw_fif(askopenfilename(title="Please choose raw file"))
    #    raw=mne.read_epochs("SavedResults/S2/det_epochs-epo.fif")
    root.destroy()
    raw.load_data()
    raw.set_montage(montage=mne.channels.make_standard_montage('biosemi256', head_size=0.089), raise_if_subset=False)
    raw.set_eeg_reference(ref_channels=['M1', 'M2'])

    print("plotting psd...")
    eog_map_dict = {'Nose': 'eeg', 'LHEOG': 'eeg', 'RHEOG': 'eeg', 'RVEOGS': 'eeg', 'RVEOGI': 'eeg', 'M1': 'eeg',
                    'M2': 'eeg', 'LVEOGI': 'eeg'}
    raw.set_channel_types(mapping=eog_map_dict)
    # raw.plot_psd(fmin=0, fmax=25, picks=range(20), n_fft=10 * 2048)
    # plt.show()
    #    ica = mne.preprocessing.read_ica(askopenfilename(title="Please choose ICA file"))
    ica = mne.preprocessing.read_ica(input("file?"))
    # print("plotting components...")
    # ica.plot_components(picks=components, show=False)
    # plt.show()
    print('creating epochs for plotting components...')

    events = mne.find_events(raw, stim_channel="Status", mask=255, min_duration=2 / 2048)
    event_dict_aud = {'short_word': 12, 'long_word': 22}
    event_dict_vis = {'short_face': 10, 'long_face': 20,
                      'short_anim': 12, 'long_anim': 22,
                      'short_obj': 14, 'long_obj': 24,
                      'short_body': 16, 'long_body': 26}
    epochs = mne.Epochs(raw, events, event_id=event_dict_vis, tmin=-0.4, baseline=(-0.25, -.10),
                        tmax=1.9, preload=True, reject_by_annotation=True)

    print("plotting properties...")
    # the beginning of each components group to be shown
    comp_jumps = np.linspace(0, ica.n_components_, int(ica.n_components_ / 8) + 1)
    for i in range(len(comp_jumps)):  # go over the components and show 8 each time
        comps = range(int(comp_jumps[i]), int(comp_jumps[i + 1]))
        print("plotting from component " + str(comps))
        plot_correlations(ica, raw, components=comps,
                          picks=['A19', 'Nose', 'RHEOG', 'LHEOG', 'RVEOGS', 'RVEOGI', 'M1', 'M2', 'LVEOGI'])

        ica.plot_properties(epochs, picks=comps, show=False, psd_args={'fmax': 100})  # plot component properties
        ica.plot_sources(epochs, picks=comps, show=False)  # plot sources
        print("plotting")
        plt.show()
        if input("keep plotting? (Y/N)") == "N":
            break
