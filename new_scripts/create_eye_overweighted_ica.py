# automatic artifact rejection and epoching for overweighted data
# %%
from sys import argv
from os.path import join
from preprocess_utilities import *
import matplotlib

matplotlib.use('Qt5Agg')

# %% params and functions
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
et_filename = f"sub-{subject_num}_task-{modality}-et.asc"
filename = f"sub-{subject_num}_task-{modality}-{low_cutoff_freq:.2f}hpf-rejections-raw.fif"
raw = mne.io.read_raw_fif(join(save_dir, filename), preload=True)
# %% add eye tracker triggers
raw = add_eytracker_triggers(raw, join(save_dir, et_filename))
# %% overweight
events = mne.find_events(raw, stim_channel="Status", mask=255, min_duration=2 / raw.info['sfreq'])
raw_for_ica, threshold_autoreject = multiply_event(raw, TRIG_DICT, events,
                                                   saccade_id=ET_TRIG_DICT["saccade"], size_new=OVERWEIGHT)
# %% add RDI annotations
raw_for_ica.annotations = raw.annotations

# %% fit ICA
ica = mne.preprocessing.ICA(n_components=.95, method='infomax',
                            random_state=97, max_iter=600, fit_params=dict(extended=True))
ica.fit(raw_for_ica, reject_by_annotation=True, reject=threshold_autoreject)
# %%
ica.save(join(save_dir, f"sub-{subject_num}_task-{modality}-{low_cutoff_freq:.2f}hpf-overweighted-ica.fif"))
raw.save(join(save_dir, filename),
         overwrite=True)  # save the raw file with the additional triggers instead of the old one
