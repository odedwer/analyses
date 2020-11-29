# TODO manual & automatic annotation of bad segments
# %%
from sys import argv
from os.path import join
from preprocess_utilities import *
import matplotlib
from EyeLinkProcessor import EyeLinkProcessor
from ParserType import ParserType
from SaccadeDetectorType import SaccadeDetectorType

RESAMPLE_FREQ = 512

CHANNELS_TO_DROP = ['ET_RX', 'ET_RY', 'ET_R_PUPIL', 'ET_LX', 'ET_LY',
                    'ET_L_PUPIL', 'Photodiode', 'ResponseBox']

MODALITY_ERR_MSG = "You did not enter a correct modality. Please attempt again:"
VALID_MODALITIES = ['visual', 'auditory']
MODALITY_MSG = "Please enter the modality (auditory/visual): "
HPF_MSG = "Please enter the higphass filter cutoff: "
SUBJECT_MSG = "Please enter the subject number: "
BASE_DATA_DIR = "../data/"

SUBJECT_NUMBER_IDX = 1
MODALIDY_IDX = 2
HIGH_PASS_IDX = 3

JUMP_CRITERIA = 1000e-6
REJ_CRITERIA = 1e-6


def read_bdf_files(preload=True):
    """
    Reads bdf file from disk. If there are several files, reads them all to different raw objects.
    GUI will open to choose a file from a folder. Any file in that folder that has the same name of the
    chosen file up to the last _ in the filename will be added opened as a raw object, by
    order of names (lexicographic)
    :return: List of the raw objects (preloaded)
    """
    # TODO: don't concatenate, return list
    # get file path using GUI
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


subject_num = get_subject_number()
modality = get_modality()
low_cutoff_freq = get_highpass_cutoff()

# upload raw files AFTER robust detrending
raw = read_bdf_files(preload=True)
raw.drop_channels(CHANNELS_TO_DROP)
copy_raw = raw.copy()  # make a copy before adding the new channel
raw = raw.resample(RESAMPLE_FREQ, n_jobs=12)

raw.filter(h_freq=None, l_freq=low_cutoff_freq, n_jobs=12)
raw.notch_filter(freqs=np.arange(50, 251, 50))
# raw.plot(n_channels=30, duration=30)  # exclude electrodes from artifact rejection
raw = set_reg_eog(raw)
time_rejected = []

for i in np.arange(50, 260, 5):
    # reject artifacts based on +- 50 ms above threshold
    raw = annotate_bads_auto(raw, reject_criteria=i * REJ_CRITERIA,
                             jump_criteria=JUMP_CRITERIA)  # by threshold and jump
    time_rejected.append(round(sum(raw._annotations.duration), 2))

subject_string = f"sub-{subject_num}"
save_dir = join(BASE_DATA_DIR, subject_string, "eeg", modality)
raw.annotations.save(
    join(save_dir, f"{subject_string}_task-{modality}-{low_cutoff_freq:.2f}hpf-rejections-annotations.fif"))
raw.save(
    join(save_dir, f"{subject_string}_task-{modality}-{low_cutoff_freq:.2f}hpf-rejections-raw.fif"))

# TODO: save the following files:
# sub-0N_task-X-1hpf-rejections-anotation.fif [the times that were annotated]
# sub-0N_task-X-1hpf-rejections-raw.fif [annotated raw file]
# where N is subject num and X is modality (visual, auditory)
