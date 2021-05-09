"""
gets the ICA file and raw files and allows inspection and rejections of components
saves the rejected components, the clean raw file and the clean epochs
"""
from sys import argv
from os.path import join
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from preprocess_utilities import *
# functions
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
TRIG_DICT = {"short_object":110,"long_object":112,
             "short_pattern":120,"long_pattern":122,
             "short_face":130,"long_face":132,
             "short_body":140,"long_body":142}
ET_TRIG_DICT = {'blink': 99, 'saccade': 98, 'fixation': 97}
OVERWEIGHT = 5  # How many times to overweight saccades
SUBJECT_NUMBER_IDX = 1
MODALIDY_IDX = 2
HIGH_PASS_IDX = 3


# %% read data
subject_num = input("subject number?")
modality = get_modality()
if(modality!="visual"):
    TRIG_DICT = {"short_word":12,"long_word":22}

low_cutoff_freq = get_highpass_cutoff()
save_dir = join(BASE_DATA_DIR, f"sub-{subject_num}", "eeg", modality)
ica_filename = f"sub-{subject_num}_task-{modality}-{low_cutoff_freq:.2f}hpf-overweighted-ica.fif"
filtered_filename = f"sub-{subject_num}_task-{modality}-{low_cutoff_freq:.2f}hpf-rejections-raw.fif"
unfiltered_filename = f"sub-{subject_num}_task-{modality}-unfiltered-raw.fif"

raw_filtered = mne.io.read_raw_fif(join(save_dir, filtered_filename), preload=True)
raw_unfiltered = mne.io.read_raw_fif(join(save_dir, unfiltered_filename), preload=True)
ica = mne.preprocessing.read_ica(join(save_dir, ica_filename))

# variance ratio
events = mne.find_events(raw_filtered, stim_channel="Status", mask=255, min_duration= 1/ raw_filtered.info['sfreq'])
ratios = fixation_saccade_variance_ratio(raw_filtered,ica,events)

# %% inspection code ,  plot properties, inspect TFR, see all topos and look at psd's together
ica.plot_sources(raw_filtered)

stimuli = ['short_scrambled', 'long_scrambled', 'short_face', 'long_face',
           'short_obj', 'long_obj', 'short_body', 'long_body']
if modality == 'auditory':
    stimuli = ['long_word', 'short_word']


epochs_filt = mne.Epochs(raw_filtered, events, event_id=TRIG_DICT,
                    tmin=-0.4, tmax=1.9, baseline=(-0.25, -0.1),
                    reject_tmin=-.1, reject_tmax=1.5,  # reject based on 100 ms before trial onset and 1500 after
                    preload=True, reject_by_annotation=True)

#%%
ica_round = 5*(int(input("enter ica round starting from 1:"))-1)
ica.plot_properties(epochs_filt, picks=np.arange(5)+ica_round,psd_args={'fmax':120})

#%% can either choose manually or write 000 and go through all of them
comp_start = f"ICA{input('From which component should we start?')}"  # from which component to start showing
ica.exclude = plot_ica_component(raw_filtered, ica, events, dict(**TRIG_DICT, **ET_TRIG_DICT), stimuli, comp_start)

#%%
ica.plot_components()
ica_raw=ica.get_sources(raw_filtered)
ch_dict={}
for name in ica_raw.ch_names:
    ch_dict[name] = "eeg"
ica_raw.set_channel_types(ch_dict)
ica_raw.plot_psd(picks=ica_raw.ch_names[10:18],n_overlap=int(0.2*raw_filtered.info['sfreq']),
                 n_fft=int(2*raw_filtered.info['sfreq']))

# %% apply solution and epoch for ERPs
ica.apply(raw_unfiltered)
raw_filt = raw_unfiltered.copy().filter(l_freq=1, h_freq=30)  # performing filtering on copy of raw data, not on raw itself or epochs
raw_filt.notch_filter([50, 100, 150])  # notch filter
raw_filt = raw_filt.interpolate_bads(mode='accurate', verbose=True)
events = mne.find_events(raw_unfiltered, stim_channel="Status", mask=255, min_duration= 2/ raw_unfiltered.info['sfreq'])

epochs_filt_clean = mne.Epochs(raw_filt, events, event_id=TRIG_DICT,
                    tmin=-0.4, tmax=1.9, baseline=(-0.25, -0.1),
                    reject_tmin=-.1, reject_tmax=1.5,  # reject based on 100 ms before trial onset and 1500 after
                    preload=True, reject_by_annotation=True)


selected_thresh, thresholds_pairs = get_rejection_threshold(epochs_filt_clean)
n_trials = len(epochs_filt_clean)
epochs_filt_clean.drop_bad(reject=selected_thresh)

#%%
print(f"removed {n_trials - len(epochs_filt_clean)} trials by peak to peak rejection with threshold {np.round(selected_thresh['eeg'],5)} V on eeg and {np.round(threshold['eog'],5)} V on eog")
epochs_filt_clean.plot()

# %% save
exclusions = pd.DataFrame({"excluded": ica.exclude})
exclusions.to_csv(join(save_dir,f"sub-{subject_num}_task-{modality}-{low_cutoff_freq:.2f}hpf-overweighted-ica-rejected.csv"))
epochs_filt_clean.save(join(save_dir,f"sub-{subject_num}_task-{modality}-1-30-bp-epo.fif"))
raw_unfiltered.save(join(save_dir,f"sub-{subject_num}_task-{modality}-unfiltered-clean-raw.fif"))