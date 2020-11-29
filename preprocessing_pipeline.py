# %
# ##import
import numpy as np
from preprocess_utilities import *
import matplotlib
from EyeLinkProcessor import EyeLinkProcessor
from ParserType import ParserType
from SaccadeDetectorType import SaccadeDetectorType
# matplotlib.use('Qt5Agg')

# %%
raw=mne.io.read_raw_fif(input(),preload=True)
raw.filter(h_freq=None, l_freq=1, n_jobs=12)
time_rejected=[]
for i in np.arange(50,260,5):
    # reject artifacts based on +- 50 ms above threshold
    raw = annotate_bads_auto(raw, reject_criteria=i*1e-6, jump_criteria=1000e-6)  # by threshold and jump
    time_rejected.append(round(sum(raw._annotations.duration), 2))

# %%
# upload raw files AFTER robust detrending
raws = read_bdf_files(preload=True)
# concatenate to one raw file
raw = raws[0]#mne.concatenate_raws(raws)
raw.drop_channels(['ET_RX', 'ET_RY', 'ET_R_PUPIL', 'ET_LX', 'ET_LY',
                   'ET_L_PUPIL', 'Photodiode', 'ResponseBox'])
copy_raw = raw.copy()  # make a copy before adding the new channel
raw = raw.resample(512,n_jobs=12)
raw.filter(h_freq=None, l_freq=1, n_jobs=12)
raw.notch_filter(freqs=np.arange(50, 251, 50))
raw.plot(n_channels=30, duration=30)  # exclude electrodes from artifact rejection
raw = set_reg_eog(raw)

# %%
raw = annotate_bads_auto(raw, reject_criteria=150e-6, jump_criteria=100e-6) #by threshold and jump
# %% plot again to see annotations and mark missed noise/jumps
raw.plot(n_channels=20, duration=10)  # to see data and mark bad  segments

# %%
print("total time annotated as bad: ", round(sum(raw._annotations.duration), 2))
# %% drop bad channels, annotate bad intervals
plot_all_channels_var(raw, max_val=4e-7, threshold=100e-10)  # max value for visualization in case of large value
raw.plot(n_channels=30, duration=30)  # to see data and mark bad channels
# %% set bads
raw.info['bads']


# %%
#if input("auditory? (Y/N)") == 'Y':
#    craw = annotate_breaks(raw)  # only for auditory

# set the montage of the electrodes - position on head
# %%
raw.set_montage(montage=mne.channels.read_custom_montage("SavedResults/S2/S2.elc"), raise_if_subset=False)
raw.set_eeg_reference()
# %%
# reject bad intervals based on peak to peak in ICA
reject_criteria = dict(eeg=200e-6, eog=300e-5)  # 300 μV and only extreme eog events
rej_step = .1  # in seconds

# %% set events
et_processor = EyeLinkProcessor("SavedResults/S2/vis_S2.asc",ParserType.MONOCULAR_NO_VELOCITY,
                                SaccadeDetectorType.ENGBERT_AND_MERGENTHALER)
et_processor.sync_to_raw(raw)
saccade_times = et_processor.get_synced_microsaccades()
blink_times = et_processor.get_synced_blinks()
fixation_times = et_processor.get_synced_fixations()
#check sync - shold see that orange markers have close blue lines from the EEG
eog_events = mne.preprocessing.find_eog_events(raw, 998)
plt.plot(np.sum([np.arange(len(raw._data[0]))==i for i in eog_events[:,0]],axis=0))# EOG channel events
plt.plot(np.in1d(np.arange(len(raw.get_data(1)[0])),blink_times),linewidth=.7) #blink triggers

#%% add triggers to data
saccade_times=np.sort(np.concatenate([saccade_times,saccade_times+1,saccade_times+2])) # make them longer
blink_times=np.sort(np.concatenate([blink_times,blink_times+1,blink_times+2])) # make them longer
fixation_times=np.sort(np.concatenate([fixation_times,fixation_times+1,fixation_times+2])) # make them longer
raw._data[raw.ch_names.index("Status")][blink_times.astype(np.int)] = 99  # set blinks
raw._data[raw.ch_names.index("Status")][saccade_times.astype(np.int)] = 98  # set saccades
raw._data[raw.ch_names.index("Status")][fixation_times.astype(np.int)] = 97  # set fixations

# %%
events = mne.find_events(raw, stim_channel="Status", mask=255, min_duration=2 / raw.info['sfreq'])
event_dict_aud = {'blink':99, 'saccade':98,'fixation':97,
                  'short_word': 12, 'long_word': 22}
event_dict_vis = {'blink':99, 'saccade':98,'fixation':97,
                  'short_scrambled': 110, 'long_scrambled': 112,
                  'short_face': 120, 'long_face': 122,
                  'short_obj': 130, 'long_obj': 132,
                  'short_body': 140, 'long_body': 142}#,#

event_dict = {'short_scrambled': 110, 'long_scrambled': 112,
              'short_face': 120, 'long_face': 122,
              'short_obj': 130, 'long_obj': 132,
              'short_body': 140, 'long_body': 142} # for cutting trials to ICA
raw_for_ica = multiply_event(raw,event_dict,events, event_id=event_dict_vis["saccade"],size_new=4)

# %% for saving the multiplied raw file:
s_num = input("subject number?")
raw_for_ica.save(f"SavedResults/S{s_num}/S{s_num}_aud_multiplied_for_ica-raw.fif")#,overwrite=True)
# %%
ica = mne.preprocessing.read_ica(input("file?"))

# %%fit ica
ica = mne.preprocessing.ICA(n_components=.95, method='infomax',
                            random_state=97, max_iter=800, fit_params=dict(extended=True))
ica.fit(raw_for_ica, reject_by_annotation=True, reject=reject_criteria)
ica.save(
    "SavedResults/S"+input("subject number?")+"/"+input("name?")+"-ica.fif")
# example: raw.save('visual-detrended-s2-rejected100-raw.fif')

# %%
ica.plot_sources(raw)
# stimuli = ['short_scrambled', 'long_scrambled','short_face', 'long_face',
#            'short_obj', 'long_obj','short_body', 'long_body']
stimuli = ['long_word','short_word']
comp_start = 0  # from which component to start showing
ica.exclude = plot_ica_component(raw, ica, events, event_dict_vis, stimuli, comp_start)

# %% prepare unfiltered data and apply ICA
copy_raw = set_reg_eog(copy_raw)
copy_raw._annotations = raw._annotations
copy_raw.info['bads'] = raw.info['bads']
copy_raw.set_montage(montage=mne.channels.read_custom_montage("SavedResults/S2/S2.elc"), raise_if_subset=False)
copy_raw.set_eeg_reference()
copy_raw._data[copy_raw.ch_names.index("Status")][blink_times.astype(np.int)] = 99  # set blinks
copy_raw._data[copy_raw.ch_names.index("Status")][saccade_times.astype(np.int)] = 98  # set saccades

# %%
#ica.exclude=[0,1,9,10,11,12,15,16,17,18,19,20,21,23,24,25,26,27,28]
ica.apply(copy_raw)
# copy_raw.save("SavedResults/S4/S4_vis_unfiltered_rejected200jump100_after_ica-raw.fif")


# %% # epoch- set triggers dictionairy, find events, crate epoch objects - divided by triggers
raw_filt = raw.copy().filter(l_freq=1, h_freq=30)  # performing filtering on copy of raw data, not on raw itself or epochs
raw_filt.notch_filter([50, 100, 150])  # notch filter


# %% epoch raw data without filtering for TF analysis
epochs = mne.Epochs(raw, events, event_id=event_dict_aud,
                    tmin=-0.4, tmax=1.9, baseline=(-0.25, -0.1),
                    reject=reject_criteria,
                    reject_tmin=-.1, reject_tmax=1.5,  # reject based on 100 ms before trial onset and 1500 after
                    preload=True, reject_by_annotation=True)
epochs_filt = mne.Epochs(raw_filt, events, event_id=event_dict_aud,
                    tmin=-0.4, tmax=1.9, baseline=(-0.25, -0.1),
                    reject=reject_criteria,
                    reject_tmin=-.1, reject_tmax=1.5,  # reject based on 100 ms before trial onset and 1500 after
                    preload=True, reject_by_annotation=True)

# %% create evokeds and plot comparison
evoked_L = epochs_filt["long_word"].average()
evoked_S = epochs_filt["short_word"].average()
mne.viz.plot_compare_evokeds({"Long":evoked_L,"Short":evoked_S},"A1",title="Long and short evoked response",vlines=[.8,1.5])# %%
ds_epochs = epochs.copy().resample(512)
# %%

# %%
def plt_plot(series):
    pass