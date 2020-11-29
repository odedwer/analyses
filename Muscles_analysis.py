from preprocess_utilities import *

raw_m = mne.io.read_raw_bdf(askopenfilename(title="Please choose "),preload=True)
raw_m.drop_channels(["A24", "C31", "D6", "D30"])  # bridged/noisy channels we choose to remove ##n
raw_m.drop_channels(['Ana' + str(i) for i in range(1, 9)])
raw_m.load_data().filter(l_freq=1., h_freq=None)  ##n
raw_m.set_montage(montage=mne.channels.make_standard_montage('biosemi256', head_size=0.089), raise_if_subset=False)
ica = mne.preprocessing.ICA(n_components=.99, random_state=97, max_iter=800)
ica.fit(raw_m)
ica.save('ica_muscles.fif')

# %%
# plot components topography
ica.plot_components(outlines='skirt', picks=range(20))
# %%
ica.plot_sources(raw_m, range(10, 20))

# %%
# plot properties of component by demand
ica.plot_properties(raw_m, picks=range(11))
