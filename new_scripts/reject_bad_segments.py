# manual & automatic annotation of bad segments
# %%
from sys import argv
from os.path import join
import matplotlib

matplotlib.use('Qt5Agg')

from EyeLinkProcessor import EyeLinkProcessor
from ParserType import ParserType
from SaccadeDetectorType import SaccadeDetectorType

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
        print(f"read parameter {arg} from CLI")
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


RESAMPLE_FREQ = 512

CHANNELS_TO_DROP = ['ET_RX', 'ET_RY', 'ET_R_PUPIL', 'ET_LX', 'ET_LY',
                     'ET_L_PUPIL', 'Photodiode', 'ResponseBox'] #[f"Ana{i}" for i in np.arange(1, 9)]  #

MODALITY_ERR_MSG = "You did not enter a correct modality. Please attempt again:"
VALID_MODALITIES = ['visual', 'auditory_w', 'auditory_b']
MODALITY_MSG = "Please enter the modality (auditory_w/auditory_b/visual): "
HPF_MSG = "Please enter the higphass filter cutoff:"
SUBJECT_MSG = "Please enter the subject number:"
BASE_DATA_DIR = "S:\Lab-Shared\Experiments\HighDenseGamma\data"
TRIG_DICT = {"short_object":110,"long_object":112,
             "short_pattern":120,"long_pattern":122,
             "short_face":130,"long_face":132,
             "short_body":140,"long_body":142}
ET_TRIG_DICT = {'blink': 99, 'saccade': 98, 'fixation': 97}
SUBJECT_NUMBER_IDX = 1
MODALIDY_IDX = 2
HIGH_PASS_IDX = 3

# %%get inputs
subject_num = input("subject num?")
# %%
modality = get_modality()
# %%
low_cutoff_freq = get_highpass_cutoff()

# %% upload raw files
raw = read_bdf_files(preload=True)
raw.set_montage(montage=mne.channels.read_custom_montage("S2.elc"), raise_if_subset=False)
raw = set_reg_eog(raw)

raw.drop_channels(CHANNELS_TO_DROP)  # default in the data that are not recorded
raw.set_eeg_reference()
raw = set_reg_eog(raw)
raw = raw.resample(RESAMPLE_FREQ, n_jobs=12)
unfiltered_raw = raw.copy()  # make a copy before filtering

raw.filter(h_freq=None, l_freq=low_cutoff_freq, n_jobs=12)
raw.notch_filter(freqs=np.arange(50, 251, 50), n_jobs=12)
plot_all_channels_var(raw, max_val=4e-7, threshold=100e-10)  # max value for visualization in case of large values

raw.plot(n_channels=60, duration=50)  # raw data inspection for marking bad electrodes and big chunks of bad data
manual_annot = raw.annotations  # saved for later in the script
unfiltered_raw.info['bads'] = raw.info['bads']
unfiltered_raw.set_annotations(raw.annotations)

raw.set_eeg_reference()

subject_string = f"sub-{subject_num}"
save_dir = join(BASE_DATA_DIR, subject_string, "eeg", modality)
# %%
raw.annotations.save(
    join(save_dir, f"{subject_string}_task-{modality}-{low_cutoff_freq:.2f}hpf-rejections-annot.fif"))
raw.save(
    join(save_dir, f"{subject_string}_task-{modality}-{low_cutoff_freq:.2f}hpf-rejections-raw.fif"),overwrite=True)
unfiltered_raw.save(
    join(save_dir, f"{subject_string}_task-{modality}-unfiltered-raw.fif"),overwrite=True)
