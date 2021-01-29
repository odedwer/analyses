"""
Gets the clean data
generates time-frequency plot, filter HFB on raw data and create HFB power epochs
returns HFB evoked, TFR, and duration tracking score per electrode
"""

from sys import argv
from os.path import join
import pandas as pd
import gc
from mne.time_frequency import tfr_morlet, tfr_multitaper, psd_multitaper, psd_welch, tfr_stockwell

import matplotlib
matplotlib.use('Qt5Agg')
from preprocess_utilities import *
# %% functions
def read_bdf_files(preload=True):
    """
    :return: List of the raw objects (preloaded)
    """
    global subject_num, modality
    subject_string = f"sub-{subject_num}"
    filename = join(BASE_DATA_DIR, subject_string, "eeg", "raw", subject_string + "_task-" + modality + "-raw.bdf")
    return mne.io.read_raw_bdf(filename, preload=preload)
def read_fif_files(preload=True):
    """
    :return: List of the raw objects (preloaded)
    """
    global subject_num, modality
    subject_string = f"sub-{subject_num}"
    filename = join(BASE_DATA_DIR, subject_string, "eeg", "raw", subject_string + "_task-" + modality + "-raw.fif")
    return mne.io.read_raw_bdf(filename, preload=preload)
def get_from_argv(idx, msg):
    if len(argv) < idx + 1:
        arg = input(msg)
    else:
        arg = argv[idx]
    return arg
def get_subject_number():
    return get_from_argv(SUBJECT_NUMBER_IDX, SUBJECT_MSG)
def get_highpass_cutoff():
    return float(get_from_argv(HIGH_PASS_IDX, HPF_MSG))
def get_modality():
    modality = get_from_argv(MODALIDY_IDX, MODALITY_MSG).lower()
    while modality not in VALID_MODALITIES:
        print(MODALITY_ERR_MSG)
        modality = input(MODALITY_MSG).lower()
    return modality

# %% params
MODALITY_ERR_MSG = "You did not enter a correct modality. Please attempt again:"
VALID_MODALITIES = ['visual', 'auditory_w', 'auditory_b']
MODALITY_MSG = "Please enter the modality (auditory/visual): "
HPF_MSG = "Please enter the higphass filter cutoff: "
SUBJECT_MSG = "Please enter the subject number: "
BASE_DATA_DIR = "S:\Lab-Shared\Experiments\HighDenseGamma\data"
TRIG_DICT = {'short_body':11, 'long_body':13,
             'short_face':21,'long_face':23,
             'short_place':31,'long_place':33,
             'short_pattern':41,'long_pattern':43,
             'short_object':51,'long_object':53}
ET_TRIG_DICT = {'blink': 99, 'saccade': 98, 'fixation': 97}
OVERWEIGHT = 5  # How many times to overweight saccades
SUBJECT_NUMBER_IDX = 1
MODALIDY_IDX = 2
HIGH_PASS_IDX = 3


# %% read data
subject_num = input("get_subject_number")
modality = get_modality()
if(modality!="visual"):
    TRIG_DICT = {"short_word":12,"long_word":22}

save_dir = join(BASE_DATA_DIR, f"sub-{subject_num}", "eeg", modality)
unfiltered_filename = f"sub-{subject_num}_task-{modality}-unfiltered-clean-raw.fif"

raw_unfiltered = mne.io.read_raw_fif(join(save_dir, unfiltered_filename), preload=True)

# %% parameters
stimuli =list(TRIG_DICT.keys())

freq_range = [5, 200]
base_correction = (-0.25, -.10)  # when epoch starts at -0.400
correction_mode = 'logratio'
time_end_short = int(modality=="visual")*0.5 + int(modality!="visual")*0.8
time_end_long = int(modality=="visual")*1.2 + int(modality!="visual")*1.5
freqsH = np.logspace(5, 7.6, 50, base=2)
freqs_all = np.logspace(1, 7.6, 40, base=2)
freqsL = np.logspace(1, 5, 40, base=2)

#%% create unfiltered epochs
events = mne.find_events(raw_unfiltered, stim_channel="Status", mask=255, min_duration= 2/ raw_unfiltered.info['sfreq'])
epochs = mne.Epochs(raw_unfiltered, events, event_id=TRIG_DICT,
                    tmin=-0.4, tmax=1.9, baseline=(-0.25, -0.1),
                    reject_tmin=-.1, reject_tmax=1.5,  # reject based on 100 ms before trial onset and 1500 after
                    preload=True, reject_by_annotation=True)
epochs.plot_image(picks=['A1'])

#%% TFR plots
power_long_H = tfr_morlet(epochs[stimuli[1::2]], freqs=freqsH, average=False,
                          n_cycles=freqsH / 10, use_fft=True,
                          return_itc=False, decim=3, n_jobs=12)
power_long_H=power_long_H.average()

power_short_H = tfr_morlet(epochs[stimuli[::2]], freqs=freqsH, average=False,
                           n_cycles=freqsH / 10, use_fft=True,
                           return_itc=False, decim=3, n_jobs=12)
power_short_H = power_short_H.average()

power_long_L = tfr_morlet(epochs[stimuli[1::2]], freqs=freqsL, average=False,
                          n_cycles=freqsL / 5, use_fft=True,
                          return_itc=False, decim=3, n_jobs=12)
power_long_L = power_long_L.average()

power_short_L = tfr_morlet(epochs[stimuli[::2]], freqs=freqsL, average=False,
                           n_cycles=freqsL / 5, use_fft=True,
                           return_itc=False, decim=3, n_jobs=12)
power_short_L = power_short_L.average()

#%% inspect
power_name=f"power_{input('short or long?')}_{input('High or low frequencies (H/L)?')}"
curr_TFR = eval(power_name).copy()
curr_TFR.plot_topo(baseline=base_correction, mode=correction_mode, title=power_name,
                       tmin=-0.3, tmax=1.8,vmin=-.25, vmax=.25, layout_scale=.5 )

# %% show ERP after hilbert
## filter raw data using iir butterwoth filter of order 3
raw_hilb = []
bands = [(58,78),(78,98),(102,122)]
nbands = len(bands)  # starting from 50, jumping by 20
for i in range(nbands):
    curr_l_freq = bands[i][0]
    curr_h_freq = bands[i][1]
    raw_bp = raw_unfiltered.copy().filter(l_freq=curr_l_freq, h_freq=curr_h_freq, method='iir',
                               iir_params=dict(order=3, ftype='butter'), n_jobs=12)
    raw_hilb.append(raw_bp.apply_hilbert(envelope=True))  # compute envelope of analytic signal
    raw_hilb[i]._data[:] = 10 * np.log10(raw_hilb[i]._data[:] ** 2)  # change to dB
    raw_hilb[i]._data = raw_hilb[i]._data[:] - np.reshape(raw_hilb[i]._data[:].mean(1),(raw_hilb[i].info['nchan'],1))
    # demean to apply correction
    print("finished band of " + str(curr_l_freq) + " - " + str(curr_h_freq))
    input()

##  compute mean of all filter bands
raw_hilb[0]._data = (raw_hilb[0]._data + raw_hilb[1]._data + raw_hilb[2]._data) / nbands
#raw_hilb[0].filter(l_freq=1,h_freq=30) # should I?

#%% epoch and compute duration tracking scores
epochs_hilb = mne.Epochs(raw_hilb[0], events, event_id=TRIG_DICT,
                         tmin=-0.4, tmax=1.9, baseline=None,
                         reject_tmin=-.1, reject_tmax=time_end_long+.3,
                         preload=True, reject_by_annotation=True)
epochs_hilb.apply_baseline((-.3, -.05), verbose=True)
epochs_hilb._data /= 1e-06 # for scale
epochs_HFB_L = epochs_hilb['long_word'] #['long_body','long_face','long_place','long_object','long_pattern'] #
epochs_HFB_S = epochs_hilb ['short_word'] #['short_body','short_face','short_place','short_object','short_pattern'] ##
evokedHFB_L = epochs_HFB_L.average()
evokedHFB_S = epochs_HFB_S.average()

# %%
dt_scores = [duration_tracking_new(epochs_HFB_L,epochs_HFB_S,ch,time_diff=[time_end_short+.1,time_end_long+.1])[0]
             for ch in epochs_hilb.ch_names[:-9]]; print("Done with duration tracking")
onset_resp_score = [total_onset_power(epochs_hilb,ch)[0] for ch in epochs_hilb.ch_names[:-9]]
# %%
mne.viz.plot_topomap(dt_scores,epochs_hilb.info)
#%%
mne.viz.plot_topomap(onset_resp_score,epochs_hilb.info,cmap='Blues')
#%%
mne.viz.plot_topomap(np.array(onset_resp_score)*np.array(dt_scores),epochs_hilb.info,cmap='Greens')

#%%
electrode = input("Electrode?")
times_significant_short = ttest_on_epochs(epochs_HFB_S,electrode,title="Short epochs")
#%%
times_significant_long = ttest_on_epochs(epochs_HFB_L,electrode,title="Long epochs")


#%%
plt.bar(raw_unfiltered.ch_names[0:256],dt_scores)
plt.plot(raw_unfiltered.ch_names[0:256],dt_scores)
plt.show()

#%%
mne.viz.plot_compare_evokeds({"Long":evokedHFB_L,"Short":evokedHFB_S},electrode,
                             title="Long and short HFB",vlines=[time_end_short,time_end_long])

#%% save
epochs_HFB_L.save(join(save_dir,f"sub-{subject_num}_task-{modality}-long-58-122-hfb-epo.fif"),overwrite=True)
epochs_HFB_S.save(join(save_dir,f"sub-{subject_num}_task-{modality}-short-58-122-hfb-epo.fif"),overwrite=True)
dt_score_df = pd.DataFrame({"electrode": raw_unfiltered.ch_names[0:len(dt_scores)],"dt_score":dt_scores})
dt_score_df.to_csv(join(save_dir,f"sub-{subject_num}_task-{modality}-duration_tracking-58-122-hfb.csv"))
onset_resp_df = pd.DataFrame({"electrode": raw_unfiltered.ch_names[0:len(onset_resp_score)],"onset_resp":onset_resp_score})
onset_resp_df.to_csv(join(save_dir,f"sub-{subject_num}_task-{modality}-onset_resp-58-122-hfb.csv"))
power_short_L.save(join(save_dir,f"sub-{subject_num}_task-{modality}-short_low_freqs-tfr.fif"),overwrite=True)
power_short_H.save(join(save_dir,f"sub-{subject_num}_task-{modality}-short_high_freqs-tfr.fif"),overwrite=True)
power_long_L.save(join(save_dir,f"sub-{subject_num}_task-{modality}-long_low_freqs-tfr.fif"),overwrite=True)
power_long_H.save(join(save_dir,f"sub-{subject_num}_task-{modality}-long_high_freqs-tfr.fif"),overwrite=True)

