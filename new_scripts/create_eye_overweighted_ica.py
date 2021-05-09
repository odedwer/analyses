# automatic artifact rejection and epoching for overweighted data
# %%
from sys import argv
from os.path import join
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

# %% params and functions
MODALITY_ERR_MSG = "You did not enter a correct modality. Please attempt again:"
VALID_MODALITIES = ['visual', 'auditory_w', 'auditory_b']
MODALITY_MSG = "Please enter the modality (auditory_w/auditory_b/visual): "
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
low_cutoff_freq = get_highpass_cutoff()
save_dir = join(BASE_DATA_DIR, f"sub-{subject_num}", "eeg", modality)
et_filename = f"sub-{subject_num}_task-{modality}-et.asc"
filename = f"sub-{subject_num}_task-{modality}-{low_cutoff_freq:.2f}hpf-rejections-raw.fif"
raw = mne.io.read_raw_fif(join(save_dir, filename), preload=True)
if(modality!="visual"):
    TRIG_DICT = {"short_word":12,"long_word":22}
# %% add eye tracker triggers
raw = add_eytracker_triggers(raw, join(save_dir, et_filename))

# %% overweight
events = mne.find_events(raw, stim_channel="Status", mask=255, min_duration= 1/ raw.info['sfreq']) #MUST BE 1!! OTHERWISE MAY LOSE TRIGGERS

raw_for_ica, threshold_autoreject = multiply_event(raw, TRIG_DICT, events,
                                                   saccade_id=ET_TRIG_DICT["saccade"], size_new=OVERWEIGHT) #select a high threshold here to include blinks!

raw_for_ica.info['bads'] = raw.info['bads']
# %% fit ICA
ica = mne.preprocessing.ICA(n_components=.95, method='infomax',
                            random_state=97, max_iter=800, fit_params=dict(extended=True))
ica.fit(raw_for_ica, reject_by_annotation=True, reject=threshold_autoreject)

ica.save(join(save_dir, f"sub-{subject_num}_task-{modality}-{low_cutoff_freq:.2f}hpf-overweighted-ica.fif"))
raw.save(join(save_dir, filename),
         overwrite=True)  # save the raw file with the additional triggers instead of the old one
