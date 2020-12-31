"""
gets the ICA file and raw files and allows inspection and rejections of components
saves the rejected components, the clean raw file and the clean epochs
"""
from sys import argv
from os.path import join
import pandas as pd
from preprocess_utilities import *
import matplotlib

matplotlib.use('Qt5Agg')

# %% params
MODALITY_ERR_MSG = "You did not enter a correct modality. Please attempt again:"
VALID_MODALITIES = ['visual', 'auditory']
MODALITY_MSG = "Please enter the modality (auditory/visual): "
HPF_MSG = "Please enter the higphass filter cutoff: "
SUBJECT_MSG = "Please enter the subject number: "
BASE_DATA_DIR = "S:\Lab-Shared\Experiments\HighDenseGamma\data"
TRIG_DICT = {'short_scrambled': 110, 'long_scrambled': 112,
             'short_face': 120, 'long_face': 122,
             'short_obj': 130, 'long_obj': 132,
             'short_body': 140, 'long_body': 142}
ET_TRIG_DICT = {'blink': 99, 'saccade': 98, 'fixation': 97}
OVERWEIGHT = 5  # How many times to overweight saccades
SUBJECT_NUMBER_IDX = 1
MODALIDY_IDX = 2
HIGH_PASS_IDX = 3

# %% read data
subject_num = get_subject_number()
modality = get_modality()
low_cutoff_freq = get_highpass_cutoff()
save_dir = join(BASE_DATA_DIR, f"sub-{subject_num}", "eeg", modality)
ica_filename = f"sub-{subject_num}_task-{modality}-{low_cutoff_freq:.2f}hpf-overweighted-ica.fif"
filtered_filename = f"sub-{subject_num}_task-{modality}-{low_cutoff_freq:.2f}hpf-rejections-raw.fif"
unfiltered_filename = f"sub-{subject_num}_task-{modality}-unfiltered-raw.fif"

raw_filtered = mne.io.read_raw_fif(join(save_dir, filtered_filename), preload=True)
raw_unfiltered = mne.io.read_raw_fif(join(save_dir, unfiltered_filename), preload=True)
ica = mne.preprocessing.read_ica(join(save_dir, ica_filename))
raw_unfiltered.info['bads'] = raw_filtered.info['bads']
raw_unfiltered.annotations = raw_filtered.annotations

# %% inspect
ica.plot_sources(raw)
stimuli = ['short_scrambled', 'long_scrambled', 'short_face', 'long_face',
           'short_obj', 'long_obj', 'short_body', 'long_body']
if modality == 'auditory':
    stimuli = ['long_word', 'short_word']
comp_start = 0  # from which component to start showing
events = mne.find_events(raw_filtered, stim_channel="Status", mask=255, min_duration=2 / raw.info['sfreq'])
ica.exclude = plot_ica_component(raw_filtered, ica, events, dict(**TRIG_DICT, **ET_TRIG_DICT), stimuli, comp_start)

# %% apply solution and epoch for ERPs
ica.apply(raw_unfiltered)
raw_filt = raw_unfiltered.copy().filter(l_freq=1, h_freq=30)  # performing filtering on copy of raw data, not on raw itself or epochs
raw_filt.notch_filter([50, 100, 150])  # notch filter
epochs_filt = mne.Epochs(raw_filt, events, event_id=TRIG_DICT,
                    tmin=-0.4, tmax=1.9, baseline=(-0.25, -0.1),
                    reject_tmin=-.1, reject_tmax=1.5,  # reject based on 100 ms before trial onset and 1500 after
                    preload=True, reject_by_annotation=True)
threshold = autoreject.get_rejection_threshold(epochs_filt)
threshold['eeg'] *= 2
n_trials = len(epochs)
epochs_filt.drop_bad(reject=threshold)
print(f"removed {n_trials - len(epochs_filt)} trials by peak to peak rejection with threshold {threshold['eeg']}")
epochs_filt.plot()

# %% save
exclusions = pd.DataFrame({"excluded": ica.exclude})
exclusions.to_csv(f"sub-{subject_num}_task-{modality}-{low_cutoff_freq:.2f}hpf-overweighted-ica-rejected.csv")
epochs_filt.save(f"sub-{subject_num}_task-{modality}-1-30-bp-epo.fif")
raw_unfiltered.save(f"sub-{subject_num}_task-{modality}-unfiltered-clean-raw.fif")