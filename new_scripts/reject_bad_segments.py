# TODO manual & automatic annotation of bad segments
# %%
from sys import argv
from os.path import join
from preprocess_utilities import *
import matplotlib
from EyeLinkProcessor import EyeLinkProcessor
from ParserType import ParserType
from SaccadeDetectorType import SaccadeDetectorType

matplotlib.use('Qt5Agg')
RESAMPLE_FREQ = 512

CHANNELS_TO_DROP = [f"Ana{i}" for i in np.arange(1, 9)]  # ['ET_RX', 'ET_RY', 'ET_R_PUPIL', 'ET_LX', 'ET_LY',
# 'ET_L_PUPIL', 'Photodiode', 'ResponseBox']

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
SUBJECT_NUMBER_IDX = 1
MODALIDY_IDX = 2
HIGH_PASS_IDX = 3

subject_num = get_subject_number()
modality = get_modality()
low_cutoff_freq = get_highpass_cutoff()


# %% upload raw files
raw = read_bdf_files(preload=True)
raw.set_montage(montage=mne.channels.read_custom_montage("S2.elc"), raise_if_subset=False)
raw = set_reg_eog(raw)

raw.drop_channels(CHANNELS_TO_DROP)  # default in the data that are not recorded
raw.set_eeg_reference()
raw = set_reg_eog(raw)

copy_raw = raw.copy()  # make a copy before adding the new channel
raw = raw.resample(RESAMPLE_FREQ, n_jobs=12)

raw.filter(h_freq=None, l_freq=low_cutoff_freq, n_jobs=12)
raw.notch_filter(freqs=np.arange(50, 251, 50))
plot_all_channels_var(raw, max_val=4e-7, threshold=100e-10)  # max value for visualization in case of large values

raw.plot(n_channels=60, duration=50)  # raw data inspection for marking bad electrodes and big chunks of bad data
manual_annot = raw.annotations  # saved for later in the script

# %% interpolate bad channels
# raw_eeg = raw.copy().pick_types(meg=False, eeg=True, exclude=[])
# raw_eeg = raw_eeg.interpolate_bads(mode='fast', verbose=True)
# raw._data[0:256] = raw_eeg._data[0:256] #replace with interpolated data
raw = raw.interpolate_bads(mode='accurate', verbose=True)

subject_string = f"sub-{subject_num}"
save_dir = join(BASE_DATA_DIR, subject_string, "eeg", modality)
# %%
raw.annotations.save(
    join(save_dir, f"{subject_string}_task-{modality}-{low_cutoff_freq:.2f}hpf-rejections-annotations.fif"))
raw.save(
    join(save_dir, f"{subject_string}_task-{modality}-{low_cutoff_freq:.2f}hpf-rejections-raw.fif"))
copy_raw.save(
    join(save_dir, f"{subject_string}_task-{modality}-unfiltered-raw.fif"))

