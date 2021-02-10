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
TRIG_DICT = {'short_body':11, 'long_body':13,
             'short_face':21,'long_face':23,
             'short_place':31,'long_place':33,
             'short_pattern':41,'long_pattern':43,
             'short_object':51,'long_object':53,}
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
raw_unfiltered.info['bads'] = raw_filtered.info['bads']
raw_unfiltered._annotations = raw_filtered._annotations

# %% variance ratio
events = mne.find_events(raw_filtered, stim_channel="Status", mask=255, min_duration= 2/ raw_filtered.info['sfreq'])
ratios = fixation_saccade_variance_ratio(raw_filtered,ica,events)

# %% inspect
ica.plot_sources(raw_filtered)

stimuli = ['short_scrambled', 'long_scrambled', 'short_face', 'long_face',
           'short_obj', 'long_obj', 'short_body', 'long_body']
if modality == 'auditory':
    stimuli = ['long_word', 'short_word']
comp_start = 0  # from which component to start showing

epochs_filt = mne.Epochs(raw_filtered, events, event_id=TRIG_DICT,
                    tmin=-0.4, tmax=1.9, baseline=(-0.25, -0.1),
                    reject_tmin=-.1, reject_tmax=1.5,  # reject based on 100 ms before trial onset and 1500 after
                    preload=True, reject_by_annotation=True)
# for i in range(len(ica._ica_names)):
#     ica.plot_properties(epochs_filt, picks=i)
#     input(f"press enter to continue to component {i+1}")
ica.plot_components()

ica.exclude = plot_ica_component(raw_filtered, ica, events, dict(**TRIG_DICT, **ET_TRIG_DICT), stimuli, comp_start)

# %% apply solution and epoch for ERPs
ica.apply(raw_unfiltered)
raw_filt = raw_unfiltered.copy().filter(l_freq=1, h_freq=30)  # performing filtering on copy of raw data, not on raw itself or epochs
raw_filt.notch_filter([50, 100, 150])  # notch filter
events = mne.find_events(raw_unfiltered, stim_channel="Status", mask=255, min_duration= 2/ raw_unfiltered.info['sfreq'])

epochs_filt_clean = mne.Epochs(raw_filt, events, event_id=TRIG_DICT,
                    tmin=-0.4, tmax=1.9, baseline=(-0.25, -0.1),
                    reject_tmin=-.1, reject_tmax=1.5,  # reject based on 100 ms before trial onset and 1500 after
                    preload=True, reject_by_annotation=True)


threshold = autoreject.get_rejection_threshold(epochs_filt_clean)
threshold['eeg'] *= 2
threshold['eog'] *= 3
n_trials = len(epochs_filt_clean)
epochs_filt_clean.drop_bad(reject=threshold)

#%%local autoreject

n_interpolates = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)

from mne.utils import check_random_state  # noqa
from mne.datasets import sample  # noqa
from autoreject import (AutoReject, set_matplotlib_defaults)  # noqa
check_random_state(42)
picks = mne.pick_types(raw.info, eeg=True, stim=False, eog=False,
                       include=[], exclude=[])
ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
                thresh_method='bayesian_optimization', random_state=42, n_jobs=12)

ar.fit(epochs)
epochs_clean = ar.transform(epochs)
evoked_clean = epochs_clean.average()
evoked = epochs.average()
ar.get_reject_log(epochs).plot()


#%%
print(f"removed {n_trials - len(epochs_filt_clean)} trials by peak to peak rejection with threshold {np.round(threshold['eeg'],5)} V on eeg and {np.round(threshold['eog'],5)} V on eog")
epochs_filt_clean.plot()

# %% save
exclusions = pd.DataFrame({"excluded": ica.exclude})
exclusions.to_csv(join(save_dir,f"sub-{subject_num}_task-{modality}-{low_cutoff_freq:.2f}hpf-overweighted-ica-rejected.csv"))
epochs_filt_clean.save(join(save_dir,f"sub-{subject_num}_task-{modality}-1-30-bp-epo.fif"))
raw_unfiltered.save(join(save_dir,f"sub-{subject_num}_task-{modality}-unfiltered-clean-raw.fif"))