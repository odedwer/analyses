# %%
import pickle
import typing as tp
from datetime import datetime
from tkinter import *

import autoreject
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sn
from h5py import File
from pandas import DataFrame
from scipy import stats
from EyeLinkProcessor import EyeLinkProcessor
from ParserType import ParserType
from SaccadeDetectorType import SaccadeDetectorType


def save_data(obj, filename):
    """
    Saves data to be loaded from disk using pickle
    :param obj: The object to save
    :param filename: The name of the file in which the object is saved
    :return: None
    """
    print("Saving", filename)
    with open(filename, 'wb') as save_file:
        pickle.dump(obj, save_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename):
    """
    Load data from given file using pickle
    :param filename: The name of the file to load from
    :return: The loaded object
    """
    print("Loading", filename)
    with open(filename, 'rb') as save_file:
        return pickle.load(save_file)


#
# def read_bdf_files(preload=True):
#     """
#     Reads bdf file from disk. If there are several files, reads them all to different raw objects.
#     GUI will open to choose a file from a folder. Any file in that folder that has the same name of the
#     chosen file up to the last _ in the filename will be added opened as a raw object, by
#     order of names (lexicographic)
#     :return: List of the raw objects (preloaded)
#     """
#     # don't concatenate, return list
#     # get file path using GUI
#     Tk().withdraw()
#     filename = askopenfilename(title="Please choose ")
#     filenames = []
#     dir_path = os.path.dirname(filename)
#     just_filename = os.path.basename(filename)
#     count = 0
#     for cur_filename in os.listdir(dir_path):
#         if just_filename[:just_filename.rfind('_')] == cur_filename[:cur_filename.rfind('_')] and cur_filename[
#                                                                                                   -3:] == "bdf":
#             filenames.append(cur_filename)
#             count += 1
#     print("Found", count, " BDF files.")
#     # open file as RAW object, preload data into memory
#     ret = []
#     for file_name in sorted(filenames):
#         if ret is not None:
#             ret.append(mne.io.read_raw_bdf(os.path.join(dir_path, file_name), preload=preload))
#     return ret


def set_reg_eog(raw, add_channels=None) -> mne.io.Raw:
    """
    :param raw: the raw input we want to set eog channels in
    :param add_channels: names of channels we would like to add
    :return: raw file with eog channels marked
    """
    if add_channels is None:
        add_channels = []
    ana_map_dict = {}
    eog_map_dict = {'Nose': 'eog', 'LHEOG': 'eog', 'RHEOG': 'eog',
                    'RVEOGS': 'eog', 'RVEOGI': 'eog', 'M1': 'eog',
                    'M2': 'eog', 'LVEOGI': 'eog'}
    if len(add_channels) > 0:
        for i in add_channels:
            eog_map_dict[i] = 'eog'
    raw.set_channel_types(mapping=eog_map_dict)
    return raw


def add_bipolar_derivation(raw: mne.io.Raw, ch_1: str, ch_2: str) -> mne.io.Raw:
    """
    adds a channel to the given raw instance that is ch_1-ch_2
    :param raw: raw object to derive channel from
    :param ch_1: anode
    :param ch_2: cathode
    """
    raw = mne.set_bipolar_reference(raw, ch_1, ch_2, drop_refs=False)
    return raw


def process_epochs(trigger, epochs: mne.Epochs, notch_list=None, high_filter=30,
                   low_filter=1, samp_rate=2048):
    """
    Gal Chen this function is responsible for the processing of existing
    'epochs' object and adding the relevant filters most parameters are
    default but can be changed notch list is a list of amplitudes of line
    noise to be filtered ouy obligatory: the specific trigger we epoch by (
    "short words") and epochs object that was previously created
    """
    if notch_list is None:
        notch_list = [50]
    curr_epochs = epochs[trigger]
    filt_epochs = curr_epochs.copy()
    filt_epochs = mne.filter.notch_filter(filt_epochs, samp_rate, notch_list)
    filt_epochs.filter(l_freq=low_filter, h_freq=high_filter)
    return filt_epochs


def load_raws_from_mat(mat_filename, raws) -> list:
    """
    Reads a single .mat file to mne.io.Raw objects
    :param mat_filename: The name of the /mat file to load from.
        This function assumes that the .mat file contains only one variable.
        This variable should be a cell array containing the detrended data for each block
        in each cell.
    :param raws: The original raw objects that correspond to the data in each cell of the cell array in the given .mat file.
        These are needed for the info object in order to turn the arrays to mne.io.Raw objects
    :return: a list of raw objects that contain the data from the .mat file
    """
    print("starting....")
    arrays = list()
    with File(mat_filename) as mat_file:
        print("opened file...")
        for key in mat_file.keys():
            if key == '#refs#':
                continue
            data = mat_file[key]
            for arr in data:
                arrays.append(mat_file[arr[0]])
            for i, arr in enumerate(arrays):
                print("parsing block", str(i) + "...")
                arrays[i] = mne.io.RawArray(arr, raws[i].info)
    return arrays


def plot_correlations(ica, raw, components, picks=None) -> None:
    """
       Reads ica and raw and prints correlation matrix of all ica components and electrodes listed.
       :param ica: the ica object
       :param raw: the raw data to check correlations with
       :param picks: the electrodes from raw we want to include in the matrix
       prints correlation matrix of all listed channels, and psds of components chosen
       """
    if picks is None:
        picks = ['A1', 'Nose', 'RHEOG', 'LHEOG', 'RVEOGS', 'RVEOGI',
                 'M1', 'M2', 'LVEOGI']
    print("correlation matrix of electrodes and components...")
    data = {}
    data_electrodes = {}
    data_ica = {}
    # add raw channels
    for i in picks:
        data[i] = raw.get_data(picks=i)[0]
        data_electrodes[i] = raw.get_data(picks=i)[0]

    ica_raw: mne.io.Raw = ica.get_sources(raw)
    set_type = {i: 'eeg' for i in ica_raw.ch_names}  # setting ica_raw
    ica_raw.set_channel_types(mapping=set_type)
    for i in list(components):
        data[ica_raw.ch_names[i]] = ica_raw.get_data(picks=i)[0]
        data_ica[ica_raw.ch_names[i]] = ica_raw.get_data(picks=i)[0]

    df = DataFrame(data)
    df_electrodes = DataFrame(data_electrodes)
    df["Radial eog"] = -df_electrodes['A19'] + (df_electrodes['RHEOG'] +
                                                df_electrodes['LHEOG'] +
                                                df_electrodes['RVEOGS'] +
                                                df_electrodes['RVEOGI'] +
                                                df_electrodes['LVEOGI']) / 5
    df_electrodes["Radial eog"] = df["Radial eog"]
    df_ica = DataFrame(data_ica)
    corr_matrix = df.corr().filter(df_electrodes.columns, axis=1).filter(
        df_ica.columns, axis=0)
    # sn.set_palette(sn.color_palette('RdBu_r',11))
    sn.heatmap(corr_matrix, annot=True, vmin=-1,
               vmax=1)  # cmap=sn.color_palette('RdBu_r', 11)
    # ('red', 'green', 'blue', 'purple', 'gold', 'silver', 'black', 'brown')
    ica_raw.plot_psd(fmin=0, fmax=250, picks=components, n_fft=10 * 2048,
                     show=False, spatial_colors=False)


def plot_all_channels_var(raw, max_val, threshold, remove_from_top=8) -> None:
    """
    plotting the variance by channel for selecting bad channels
    :param raw: raw file
    :param threshold: color line in this number
    :param max_val: if any of the variances is larger than this value, reduce it for visualization. in case of extreme values
    :param remove_from_top: number of channels to remove from the last one. default is 16 to not include analog and face channels
    """
    channs = range(len(raw.ch_names) - remove_from_top - 1)
    data = raw.get_data(picks=channs)
    bad_points = raw._annotations.onset
    timepoints = np.round(raw._times[np.arange(len(data[1,]))], 2)
    not_bad_points = ~np.in1d(timepoints, np.round(bad_points, 2))
    var_vec = np.var(data[channs][:, not_bad_points],
                     axis=1)  # get only the ppoints that were not annotated
    # as bad for the variance calculation!
    var_vec[var_vec > max_val] = max_val  # for visualiztions
    electrode_letter = [i[0] for i in raw.ch_names[0:(len(channs))]]
    for i in channs:  # print names of noisy electrodes
        if var_vec[i] > threshold:
            print(raw.ch_names[i])
    colors = {'A': 'brown', 'B': 'red',
              'C': 'orange', 'D': 'gold',
              'E': 'green', 'F': 'blue',
              'G': 'pink', 'H': 'black'}
    plt.bar(x=raw.ch_names[0:(len(channs))], height=var_vec,
            color=[colors[i] for i in electrode_letter])
    plt.axhline(y=threshold, color='grey')
    plt.ylabel("Variance")
    plt.xlabel("channel")
    plt.show()


def annotate_bads_auto(raw, reject_criteria, jump_criteria,
                       reject_criteria_blink=200e-6) -> mne.io.Raw:
    """
    reads raw object and annotates automatically by threshold criteria - lower or higher than value.
    Also reject big jumps.
    supra-threshold areas are rejected - 50 ms to each side of event
    returns the annotated raw object and print times annotated
    :param jump_criteria: number - the minimum point to-point difference for rejection
    :param raw: raw object
    :param reject_criteria: number - threshold
    :param reject_criteria_blink: number, mark blinks in order to not reject them by mistake
    :return: annotated raw object
    """
    data = raw.get_data(picks='eeg')  # matrix size n_channels X samples
    del_arr = [raw.ch_names.index(i) for i in raw.info['bads']]
    data = np.delete(data, del_arr, 0)  # don't check bad channels
    block_end = (raw.get_data(264).astype(np.int) & 255)[0] == 254
    jumps = ((block_end[:-1]) |
             (abs(sum(np.diff(data))) > len(
                 data) * jump_criteria))  # mark large changes (mean change
    # over jump threshold)
    jumps = np.append(jumps, False)  # fix length for comparing

    # reject big jumps and threshold crossings, except the beggining and the
    # end.
    rejected_times = (sum(abs(data) > reject_criteria) == 1) & \
                     ((raw._times > 0.1) & (raw._times < max(raw._times) - 0.1))
    event_times = raw._times[rejected_times]  # collect all times of
    # rejections except first and last 100ms
    plt.plot(rejected_times)
    extralist = []
    data_eog = raw.get_data(picks='eog')
    eye_events = raw._times[sum(abs(data_eog) > reject_criteria_blink) > 0]
    plt.plot(sum(abs(data_eog) > reject_criteria_blink) > 0)
    plt.title(
        "blue-annotations before deleting eye events. orange - only eye events")
    # print("loop length:", len(event_times))
    for i in range(2, len(
            event_times)):  # don't remove adjacent time points or blinks
        if i % 300 == 0: print(i)
        if ((event_times[i] - event_times[i - 1]) < .05) | \
                (np.any(abs(event_times[
                                i] - eye_events) < .3) > 0):  ## if a blink occured 300ms before or after
            extralist.append(i)
    event_times = np.delete(event_times, extralist)
    event_times = np.append(event_times, raw._times[jumps])  # add jumps
    onsets = event_times - 0.05
    # print("100 ms of data rejected in times:\n", onsets)
    durations = [0.1] * len(event_times)
    descriptions = ['BAD_data'] * len(event_times)
    annot = mne.Annotations(onsets, durations, descriptions,
                            orig_time=raw.info['meas_date'])
    raw.set_annotations(annot)
    return raw


def annotate_breaks(raw, trig=254, samp_rate=2048) -> mne.io.Raw:
    """
       Reads raw and  annotates, for every start trigger, the parts from the trigger
       up to 1sec before the next one. RUN BEFORE ICA, and make sure that reject by annotation in ica is True.
       :param raw: the raw data to check correlations with
       :param trig: trigger to remove, default is 254
       :param samp_rate:
       :return: raw with annotated breaks
       """
    events = mne.find_events(raw, stim_channel="Status", mask=255)
    event_times = [i[0] / samp_rate for i in events if
                   i[2] == trig]  # time of beginning of record
    next_trig_dur = [
        (events[i + 1][0] / samp_rate - 2 - events[i][0] / samp_rate)
        for i in range(len(events) - 2) if
        events[i][2] == trig]  ##2 seconds before next (real) trigger after 254
    raw._annotations = mne.Annotations(event_times, next_trig_dur, 'BAD')
    return raw


def fixation_saccade_variance_ratio(raw, ica, events,
                                    cut_before_event=10 / 1000,
                                    cut_after_event_sac=50 / 1000,
                                    cut_after_event_fix=350 / 1000,
                                    et_trig_dict=None) -> dict:
    """
    Create the projected ica sources and epoch them by saccades and fixation.
    calculate the overall variance of both epochs for each component and plot
    their ratio, with a line marking 1.1 on the graph. Will be used to decide
    on components exclusion like Dimigen (2020, Neuroimage).

    :param raw: the filtered raw data
    :param ica: the ica file calculated on overweighted data
    :param events: events files after adding ET triggers
    :param et_trig_dict: the trigger names
    :return: variance per channel during fixation/during saccades
    """
    if et_trig_dict is None:
        et_trig_dict = {'blink': 99, 'saccade': 98,
                        'fixation': 97}
    source_raw = ica.get_sources(raw)
    source_raw._annotations = raw._annotations

    epochs_saccade = mne.Epochs(source_raw, events,
                                event_id=et_trig_dict["saccade"],
                                tmin=-cut_before_event,
                                tmax=cut_after_event_sac,
                                baseline=(
                                    -cut_before_event, cut_after_event_sac),
                                reject_tmin=-cut_before_event,
                                reject_tmax=cut_after_event_sac,
                                preload=True, reject_by_annotation=True)
    epochs_fixation = mne.Epochs(source_raw, events,
                                 event_id=et_trig_dict["fixation"],
                                 tmin=-cut_before_event,
                                 tmax=cut_after_event_fix,
                                 baseline=(
                                     -cut_before_event, cut_after_event_fix),
                                 reject_tmin=-cut_before_event,
                                 reject_tmax=cut_after_event_fix,
                                 preload=True, reject_by_annotation=True)

    data_fix = np.hstack(epochs_fixation.get_data())
    print("Shape of fixtation data:", data_fix.shape)
    raw_fix = mne.io.RawArray(data_fix, source_raw.info)

    data_sac = np.hstack(epochs_saccade.get_data())
    print("Shape of saccade data:", data_sac.shape)
    raw_sac = mne.io.RawArray(data_sac, source_raw.info)

    outlier_fix = np.quantile(np.abs(data_fix), 0.98)
    outlier_sac = np.quantile(np.abs(data_fix), 0.98)
    fixation_var = stats.tvar(data_fix, (-outlier_fix, outlier_fix), axis=1)
    saccade_var = stats.tvar(data_sac, (-outlier_sac, outlier_sac), axis=1)

    var_ratio = saccade_var / fixation_var
    plt.bar(x=source_raw.ch_names, height=var_ratio, color="darkblue")
    plt.axhline(y=1.1, color='red')
    plt.ylabel("Variance ratio fix/sac")
    plt.xlabel("ICA component")
    plt.show()
    raw_fix.plot(title="fixation epochs")
    raw_sac.plot(title="saccade epochs")
    return dict(zip(source_raw.ch_names, var_ratio))

def plot_ica_component(raw, ica, events, event_dict, stimuli, comp_start):
    """
    plot a component -
    trial, saccade, blink evoked response
    trial, saccade, blink time-frequency plots
    Topography from 3 angles
    Correlation with all eog channels
    Time course of the component (and option to zoom in and out)
    Component spectrum (psd)

    :param saccade_times: vector in the length of eeg signal with 1 on each place of saccade onset
    :param blink_times: vector in the length of eeg signal with 1 on each place of blink onset
    :param raw: raw object
    :param ica: saved ica object
    :return:
    """
    import matplotlib
    # matplotlib.use('TkAgg', warn=False, force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from mne.time_frequency import tfr_morlet
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    # Implement the default Matplotlib key bindings.
    from matplotlib.backend_bases import key_press_handler

    # Seperated out config of plot to just do it once
    def config_plot():
        fig, ax = plt.subplots(2, 3)
        return (fig, ax)

    class MatplotlibSwitchGraphs:
        def __init__(self, master, raw, ica, epochs, comp_start):
            self.master = master
            self.stimuli = stimuli
            self.raw = raw
            self.ica = ica
            self.frame = Frame(self.master)
            self.fig, self.ax = config_plot()
            self.graphIndex = comp_start
            self.maxIndex = ica.n_components_
            self.canvas = FigureCanvasTkAgg(self.fig, self.master)
            self.config_window()
            self.draw_graph(self.graphIndex)
            self.frame.pack(expand=0)
            self.epochs = epochs

        def config_window(self):
            self.canvas.mpl_connect("key_press_event", self.on_key_press)
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
            self.toolbar.update()
            self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
            self.button = Button(self.master, text="Quit", command=self._quit)
            self.button.pack(side=BOTTOM)
            self.button_exclude_comp = Button(self.master, text="Exclude",
                                              command=self.exclude)
            self.button_exclude_comp.pack(side=TOP)
            self.button_switch_up = Button(self.master, text="->",
                                           command=self.move_up)
            self.button_switch_up.pack(side=RIGHT)
            self.button_switch_down = Button(self.master, text="<-",
                                     command=self.move_down)
            self.button_switch_down.pack(side=LEFT)

        def draw_graph(self, index):
            ica_raw: mne.io.Raw = self.ica.get_sources(self.raw)
            ica_raw = ica_raw.pick(index)
            freqs = np.logspace(3.5, 7.6, 101, base=2)  # for the TF plots
            freqs_to_show = np.arange(0, len(freqs), int(len(freqs) / 10))
            # indexes of freqs to show on the graph later
            data_ica = ica_raw.get_data(picks=0)
            now = datetime.now()
            set_type = {i: 'eeg' for i in ica_raw.ch_names}  # setting ica_raw
            ica_raw.set_channel_types(mapping=set_type)
            #   self.ica.plot_properties(epochs[stimuli], picks=index, show=False,
            #                           psd_args={'fmax': 100})  # plot component properties
            # self.fig, self.ax = config_plot()
            [[self.ax[i, j].clear() for j in range(3)] for i in
             range(2)]  # clear current axes
            self.fig.suptitle(
                "Component " + str(index) + " - zoom in subplots for detail",
                fontsize=12)
            # self.ax[0, 0].plot(data_ica)
            # self.ax[0, 0].set_xlim([0, 10*ica_raw.info['sfreq']])
            # reject based on 100 ms before trial onset and 1500 after
            epochs_ica = mne.Epochs(ica_raw, events, event_id=event_dict,
                                    tmin=-0.4, tmax=1.9, baseline=(-0.25, -0.1),
                                    reject_tmin=-.1, reject_tmax=1.5,
                                    preload=True, reject_by_annotation=True)
            evoked = epochs_ica.average(picks=0)
            evoked_saccade = epochs_ica['saccade'].average(0).data
            self.ax[1, 2].plot((np.arange(len(evoked_saccade[0, :])) /
                                evoked.info['sfreq'] - 0.4),
                               evoked_saccade[0, :])
            self.ax[1, 2].axhline(0, linestyle="--", color="grey", linewidth=.6)
            self.ax[1, 2].set_title('Saccade ERP')
            self.ax[1, 2].set_ylim(-2, 2)
            self.ax[1, 2].set_ylabel('μV')

            evoked_blink = epochs_ica['blink'].average(0).data
            self.ax[1, 1].plot((np.arange(len(evoked_blink[0, :])) /
                                evoked.info['sfreq'] - 0.4),
                               evoked_blink[0, :])
            self.ax[1, 1].axhline(0, linestyle="--", color="grey", linewidth=.6)
            self.ax[1, 1].set_title('Blink ERP')
            self.ax[1, 1].set_ylabel('μV')
            self.ax[1, 1].set_ylim(-2, 2)

            # correls = [np.corrcoef(data_ica, raw._data[i])[0, 1] for i in range(len(self.raw.ch_names))]
            # self.ax[1, 0].bar(x=raw.ch_names, height=correls, color='purple')
            # self.ax[1, 0].set_title('Electrode correlation)')
            # self.ax[1, 0].set_ylabel('r')
            evoked_stimulus = epochs_ica[list(event_dict.keys())[2:]].average(
                0).data
            self.ax[1, 0].plot((np.arange(len(evoked_stimulus[0, :])) /
                                evoked.info['sfreq'] - 0.4),
                               evoked_stimulus[0, :])
            self.ax[1, 0].axhline(0, linestyle="--", color="grey", linewidth=.6)
            self.ax[1, 0].set_title('Stimulus ERP')
            self.ax[1, 0].set_ylabel('μV')
            self.ax[1, 0].set_ylim(-3, 3)
            # TF
            power_saccade = tfr_morlet(epochs_ica['saccade'], freqs=freqs,
                                       average=False,
                                       n_cycles=np.round(
                                           np.log((freqs + 13) / 10) * 10),
                                       use_fft=True,
                                       return_itc=False, picks=0, decim=3,
                                       n_jobs=12)
            TFR_s = power_saccade.average().data
            times_s = power_saccade.average().times[0:len(TFR_s[0, 0]):55]
            times_s = times_s[1:-1]
            TFR_s_corrected = (TFR_s[0].transpose() - (
                np.mean(TFR_s[0][:, 40:100], axis=1))).transpose()
            self.ax[0, 2].imshow((TFR_s_corrected[:, 55:340]), cmap='jet',
                                 origin='lowest', aspect='auto')
            self.ax[0, 2].set_title('Saccade-locked TF')
            self.ax[0, 2].set_ylabel('Hz')
            self.ax[0, 2].set_xlabel('Time (s)')
            self.ax[0, 2].set_yticks(list(freqs_to_show))
            self.ax[0, 2].set_yticklabels(np.round(freqs[freqs_to_show]))
            time_vec = np.arange(len(TFR_s[0, 0]))[0:len(TFR_s[0, 0]):55]
            time_vec = time_vec[1:-1] - 55
            self.ax[0, 2].set_xticks(list(time_vec))
            self.ax[0, 2].set_xticklabels(np.round(times_s, 1))

            power_trial = tfr_morlet(epochs_ica[list(event_dict.keys())[2:]],
                                     freqs=freqs, average=False,
                                     n_cycles=np.round(
                                         np.log((freqs + 13) / 10) * 10),
                                     use_fft=True,
                                     return_itc=False, picks=0, decim=3,
                                     n_jobs=12)
            TFR_t = power_trial.average().data
            times_t = power_trial.average().times[
                      0:len(power_saccade.average().times):55]
            times_t = times_t[1:-1]
            TFR_t_corrected = (TFR_t[0].transpose() - (
                np.mean(TFR_t[0][:, 40:100], axis=1))).transpose()
            self.ax[0, 0].imshow((TFR_t_corrected[:, 55:340]), cmap='jet',
                                 origin='lowest', aspect='auto')
            self.ax[0, 0].set_title('Stimulus-locked TF')
            self.ax[0, 0].set_ylabel('Hz')
            self.ax[0, 0].set_xlabel('Time (s)')
            self.ax[0, 0].set_yticks(list(freqs_to_show))
            self.ax[0, 0].set_yticklabels(np.round(freqs[freqs_to_show]))
            time_vec = np.arange(len(TFR_t[0, 0]))[0:len(TFR_t[0, 0]):55]
            time_vec = time_vec[1:-1] - 55
            self.ax[0, 0].set_xticks(list(time_vec))
            self.ax[0, 0].set_xticklabels(np.round(times_t, 1))

            power_blink = tfr_morlet(epochs_ica['blink'], freqs=freqs,
                                     average=False,
                                     n_cycles=np.round(
                                         np.log((freqs + 13) / 10) * 10),
                                     use_fft=True,
                                     return_itc=False, picks=0, decim=3,
                                     n_jobs=12)
            TFR_b = power_blink.average().data
            times_b = power_blink.average().times[
                      0:len(power_blink.average().times):55]
            times_b = times_b[1:-1]
            TFR_b_corrected = (TFR_b[0].transpose() - (
                np.mean(TFR_b[0][:, 40:100], axis=1))).transpose()
            self.ax[0, 1].imshow((TFR_b_corrected[:, 55:340]), cmap='jet',
                                 origin='lowest', aspect='auto')
            self.ax[0, 1].set_title('Blink-locked TF')
            self.ax[0, 1].set_ylabel('Hz')
            self.ax[0, 1].set_xlabel('Time (s)')
            self.ax[0, 1].set_yticks(list(freqs_to_show))
            self.ax[0, 1].set_yticklabels(np.round(freqs[freqs_to_show]))
            self.ax[0, 1].set_xticks(list(time_vec))
            self.ax[0, 1].set_xticklabels(np.round(times_b, 1))

            # self.ax[1, 0].plot()
            # self.ax[1, 0].set_title('Axis [1, 0]')
            # self.ax[1, 1].plot()
            # self.ax[1, 1].set_title('Axis [1, 1]')
            # self.ax.set(title="component " + str(index))
            self.canvas.draw()
            print("drawing graph took ", datetime.now() - now)

        def on_key_press(self, event):
            print("you pressed {}".format(event.key))
            key_press_handler(event, self.canvas, self.toolbar)

        def _quit(self):
            self.master.quit()  # stops mainloop

        def exclude(self):
            print("ICA component", self.graphIndex, "excluded")
            self.ica.exclude.append(self.graphIndex)

        def move_up(self):
            # Need to call the correct draw, whether we're on graph one or two
            self.graphIndex = (self.graphIndex + 1)
            if self.graphIndex > self.maxIndex:
                self.graphIndex = self.maxIndex
            self.draw_graph(self.graphIndex)

        def move_down(self):
            # Need to call the correct draw, whether we're on graph one or two
            self.graphIndex = (self.graphIndex - 1)
            if self.graphIndex < 0:
                self.graphIndex = 0
            self.draw_graph(self.graphIndex)

    root = Tk()
    # reject based on 100 ms before trial onset and 1500 after
    epochs = mne.Epochs(raw, events, event_id=event_dict,
                        tmin=-0.4, tmax=1.9, baseline=(-0.25, -0.1),
                        reject_tmin=-.1, reject_tmax=1.5,
                        preload=True, reject_by_annotation=True)
    ica.exclude = MatplotlibSwitchGraphs(root, raw, ica, epochs,
                                         comp_start).ica.exclude
    root.mainloop()
    root.destroy()
    return ica.exclude


def ica_checker(raw, ica):
    from tkinter.filedialog import askopenfilename
    import matplotlib as mpl
    mpl.use("tkAgg")
    root = Tk()
    root.withdraw()
    raw = mne.io.read_raw_fif(askopenfilename(title="Please choose raw file"))
    #    raw=mne.read_epochs("SavedResults/S2/det_epochs-epo.fif")
    root.destroy()
    raw.load_data()
    raw.set_montage(montage=mne.channels.make_standard_montage('biosemi256',
                                                               head_size=0.089),
                    raise_if_subset=False)
    raw.set_eeg_reference(ref_channels=['M1', 'M2'])

    print("plotting psd...")
    eog_map_dict = {'Nose': 'eeg', 'LHEOG': 'eeg', 'RHEOG': 'eeg',
                    'RVEOGS': 'eeg', 'RVEOGI': 'eeg', 'M1': 'eeg',
                    'M2': 'eeg', 'LVEOGI': 'eeg'}
    raw.set_channel_types(mapping=eog_map_dict)
    # raw.plot_psd(fmin=0, fmax=25, picks=range(20), n_fft=10 * 2048)
    # plt.show()
    #    ica = mne.preprocessing.read_ica(askopenfilename(title="Please choose ICA file"))
    ica = mne.preprocessing.read_ica(input("file?"))
    # print("plotting components...")
    # ica.plot_components(picks=components, show=False)
    # plt.show()
    print('creating epochs for plotting components...')

    events = mne.find_events(raw, stim_channel="Status", mask=255,
                             min_duration=2 / 2048)
    event_dict_aud = {'short_word': 12, 'long_word': 22}
    event_dict_vis = {'short_face': 10, 'long_face': 20,
                      'short_anim': 12, 'long_anim': 22,
                      'short_obj': 14, 'long_obj': 24,
                      'short_body': 16, 'long_body': 26}
    epochs = mne.Epochs(raw, events, event_id=event_dict_vis, tmin=-0.4,
                        baseline=(-0.25, -.10),
                        tmax=1.9, preload=True, reject_by_annotation=True)

    print("plotting properties...")
    # the beginning of each components group to be shown
    comp_jumps = np.linspace(0, ica.n_components_,
                             int(ica.n_components_ / 8) + 1)
    for i in range(
            len(comp_jumps)):  # go over the components and show 8 each time
        comps = range(int(comp_jumps[i]), int(comp_jumps[i + 1]))
        print("plotting from component " + str(comps))
        plot_correlations(ica, raw, components=comps,
                          picks=['A19', 'Nose', 'RHEOG', 'LHEOG', 'RVEOGS',
                                 'RVEOGI', 'M1', 'M2', 'LVEOGI'])

        ica.plot_properties(epochs, picks=comps, show=False,
                            psd_args={'fmax': 100})  # plot component properties
        ica.plot_sources(epochs, picks=comps, show=False)  # plot sources
        print("plotting")
        plt.show()
        if input("keep plotting? (Y/N)") == "N":
            break


def multiply_event(raw, event_dict, events, saccade_id=98,
                   cut_before_event=30 / 1000, cut_after_event=50 / 1000,
                   cut_epochs=1.7, size_new=1) -> tp.Tuple[mne.io.Raw, dict]:
    """
    The function creates a new raw data for ICA,
     with the extension being a multiplication of allwanted events, in order to create dominant components in the ICA.
     Stages:
     1. Cut all data between (-100)-1700 after stimulus onset
     2. Cut all data between 30ms before to 50 ms after every saccade onset and create new raw file
     3. Add to one raw file

    :param cut_epochs: how much time to take after stim onset
    :param cut_after_event: how much to cut before event
    :param cut_before_event: how much to cut after event
    :param raw: original raw file
    :param saccade_id: the name of the event to cut (saccades usually)
    :param event_dict: numbers of stimulus onsets (will be at the dictionary of events)
    :param size_new: integer. how many times should i multiply the raws list
    :return: the new concatenated raw file
    """

    raw.load_data()
    epochs_saccades = mne.Epochs(raw, events, event_id=saccade_id,
                                 tmin=-cut_before_event, tmax=cut_after_event,
                                 baseline=(-cut_before_event, cut_after_event),
                                 reject_tmin=-cut_before_event,
                                 reject_tmax=cut_after_event,
                                 # reject based on 100 ms before trial onset and 1500 after
                                 preload=True,
                                 reject_by_annotation=True)  # currently includes mean-centering - should we?
    epochs_saccades.plot()
    data_s = np.hstack(epochs_saccades.get_data())
    data_s = np.hstack([data_s.copy() for _ in range(size_new)])
    print("Shape of saccades data:", data_s.shape)
    raw_saccades = mne.io.RawArray(data_s, raw.info)

    epochs_trials = mne.Epochs(raw, events, event_id=event_dict,
                               tmin=-.1, tmax=cut_epochs,
                               baseline=(-.1, cut_epochs), reject_tmin=0,
                               reject_tmax=cut_epochs,
                               # reject based time between trial onset and 1500 after
                               preload=True,
                               reject_by_annotation=True)  # currently includes mean-centering - should we?

    threshold = autoreject.get_rejection_threshold(epochs_trials)
    threshold['eeg'] *= 2
    threshold['eog'] *= 3
    n_trials = len(epochs_trials)
    epochs_trials.drop_bad(reject=threshold)
    print(
        f"removed {n_trials - len(epochs_trials)} trials by peak to peak rejection with threshold {threshold['eeg']}")
    epochs_trials.plot()
    data_t = np.hstack(epochs_trials.get_data())

    print("Shape of trials data:", data_t.shape)
    raw_trials = mne.io.RawArray(data_t, raw.info)
    raw_for_ica = mne.concatenate_raws([raw_trials, raw_saccades])
    raw_multiplied = raw_for_ica.copy()
    # for i in range(size_new - 1):
    #     raw_multiplied = mne.concatenate_raws([raw_multiplied, raw_for_ica])
    #     print(f"length is multiplied by {i + 2}")

    return raw_multiplied, threshold


def duration_tracking(epo_A, epo_B, time_diff, p_thresh=0.01) -> np.array:
    """
    Calculates duration tracking score by point-by point t-test. after deriving t and p-value, get
    :return:dt_scores - the mean of above threshold t-tests.
    """

    samples_of_int = (epo_A.times > time_diff[0]) & (epo_A.times < time_diff[
        1])  # take only the timepoints of interest
    n_points = sum(
        samples_of_int)  # number of points of interest for later normaliztion of DT scores
    t_tests = stats.ttest_ind(epo_A.get_data(picks='eeg'),
                              epo_B.get_data(picks='eeg'))
    t_vals = t_tests.statistic[:,
             samples_of_int]  # size N_electrodes & N_timepoints - subset for timepoints of interest
    p_vals = t_tests.pvalue[:, samples_of_int]
    p_vals_sig = p_vals < p_thresh
    dt_scores = np.sum(t_vals * p_vals_sig,
                       axis=1) / n_points  # send insignificant t's to zero for the summation

    return dt_scores

def duration_tracking_new(epo_A, epo_B, channel,time_diff, cluster_thresh_t=1,nperm=2000, alpha=0.05):
    """
    A channel is defined as duration tracking and gets a score only if there is a signficant cluster between the epochs
     in the relevant time (end of short to end of long)
    Calculates duration tracking score by averaging over all point-by point t-test int the relevant time. after deriving t and p-value, get
    :return:dt_scores - the mean of point by poiny t-tests, p-values of clusters.
    """

    samples_of_int = (epo_A.times > time_diff[0]) & (epo_A.times < time_diff[1])  # take only the timepoints of interest
    epochs_perm_A = epo_A._data[:, epo_A.ch_names.index(channel), samples_of_int]
    epochs_perm_B = epo_B._data[:, epo_B.ch_names.index(channel), samples_of_int]
    T_obs, clusters, cluster_p_values, H0 = \
        mne.stats.permutation_cluster_test([epochs_perm_A,epochs_perm_B],
                                                 n_permutations=nperm, seed=1,
                                                 threshold=cluster_thresh_t, tail=1,
                                                 out_type='mask', verbose='ERROR')
    dt_score = np.mean(T_obs) #* (sum(cluster_p_values < alpha) > 0)  # send to zero if no significant cluster
    return dt_score, cluster_p_values


def add_eytracker_triggers(raw: mne.io.Raw, et_file) -> mne.io.Raw:
    """
    :string et_file: the path to eye tracker file
    raw: the raw file which the ET file matches
    :return: the raw file with the new triggers
    """
    # %% set events
    et_processor = EyeLinkProcessor(et_file, ParserType.MONOCULAR_NO_VELOCITY,
                                    SaccadeDetectorType.ENGBERT_AND_MERGENTHALER)
    et_processor.sync_to_raw(raw)
    saccade_times = et_processor.get_synced_microsaccades()
    blink_times = et_processor.get_synced_blinks()
    fixation_times = et_processor.get_synced_fixations()
    # check sync - shold see that orange markers have close blue lines from the EEG
    eog_events = mne.preprocessing.find_eog_events(raw, 998)
    plt.plot(
        np.sum([np.arange(len(raw._data[0])) == i for i in eog_events[:, 0]],
               axis=0))  # EOG channel events
    plt.plot(np.in1d(np.arange(len(raw.get_data(1)[0])), blink_times),
             linewidth=.7)  # blink triggers
    plt.show()
    # %% add triggers to data
    saccade_times = np.sort(
        np.concatenate([saccade_times, saccade_times + 1, saccade_times + 2,
                        saccade_times + 3]))  # make them longer
    blink_times = np.sort(
        np.concatenate([blink_times, blink_times + 1, blink_times + 2,
                        blink_times + 3]))  # make them longer
    fixation_times = np.sort(
        np.concatenate(
            [fixation_times, fixation_times + 1, fixation_times + 2,
             fixation_times + 3]))  # make them longer
    raw._data[raw.ch_names.index("Status")][
        blink_times.astype(np.int)] = 99  # set blinks
    raw._data[raw.ch_names.index("Status")][
        saccade_times.astype(np.int)] = 98  # set saccades
    raw._data[raw.ch_names.index("Status")][
        fixation_times.astype(np.int)] = 97  # set fixations
    return raw

def ttest_on_epochs(epochs,channel, alpha=0.05,title="epochs"):
    """
    gets on epoch and conducts 1 sample t-test for a specific electrode to test the hypothesis the mean is higher from zero
    :param epochs: the epochs file to be tested
    :param channel: wanted electrode, str
    :param alpha: significance level, float
    :return: plots the t by time for the elctrode and returns the times in which t is significant
    """

    from mne.stats import bonferroni_correction, fdr_correction
    X = epochs.get_data()  # as 3D matrix
    X = X[:, epochs.ch_names.index(channel),
        :]  # take only one channel to get a 2D array
    T, pval = stats.ttest_1samp(X, 0)

    n_samples, n_tests = X.shape
    threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)

    reject_bonferroni, pval_bonferroni = bonferroni_correction(pval,
                                                               alpha=alpha)
    threshold_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)

    reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='indep')
    threshold_fdr = np.min(np.abs(T)[reject_fdr])
    times = 1e3 * epochs.times

    plt.close('all')
    plt.plot(times, T, 'k', label='T-stat')
    plt.title(title)
    xmin, xmax = plt.xlim()
    plt.hlines(threshold_uncorrected, xmin, xmax, linestyle='--', colors='k',
               label='p=0.05 (uncorrected)', linewidth=2)
    plt.hlines(threshold_bonferroni, xmin, xmax, linestyle='--', colors='r',
               label='p=0.05 (Bonferroni)', linewidth=2)
    plt.hlines(threshold_fdr, xmin, xmax, linestyle='--', colors='b',
               label='p=0.05 (FDR)', linewidth=2)
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("T-stat")
    plt.show()
    return ({"fdr": times[T > threshold_fdr],
             "uncorrected": times[T > threshold_uncorrected]})


def vrange(starts, stops) -> np.arange:
    """Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:
        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    l = stops - starts  # Lengths of each range.
    return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())


def raw_annotate_peak_to_peak(raw: mne.io.Raw,
                              threshold: [dict, float, int] = 40e-6,
                              trig_dict=None, reject_tmin=-0.1,
                              reject_tmax=1.5,
                              events: [np.array, None] = None) -> None:
    """
    Adds bad segments annotation to an mne.Raw object, for epochs of a given
    trigger. An epoch is annotated as bad if it's peak to peak voltage
    from tmin to tmax is larger than a given threshold
    :param raw: mne.Raw to annotate
    :param trig_dict: dict of trigger values, epochs are created
                      around these triggers
    :param threshold: dict, float or int. Default 40e-6.
                      The rejection threshold - if dict, should contain a
                      key 'eeg' whose value is the threshold
    :param reject_tmin: float, default -0.1. How many seconds before the trigger
                        to consider for peak to peak calculation
    :param reject_tmax: float, default 1.5. How many seconds after the trigger
                        to consider for peak to peak calculation
    :param events: Optional, np.array of shape (N events, 3), should be a result
                   of mne.find_events. If None (default), mne.find_events is
                   used and only events which match values in trig_dict are
                   used. If given, use all the given events.
    """
    # some input validation
    if isinstance(threshold, dict):
        if 'eeg' not in threshold:
            raise KeyError("'eeg' not a key in trig_dict")
        threshold = threshold['eeg']
    elif not isinstance(threshold, int) or isinstance(threshold, float):
        raise ValueError("Threshold must be a dictionary, float or an int!")
    else:
        threshold = float(threshold)
    if not events:
        # if events are not given, take only events that are in trig_dict
        events = mne.find_events(raw, mask=255, mask_type='and')
        if trig_dict is not None:
            events = events[np.isin(events[:, 2], list(trig_dict.values())), :]
    # calculate number of samples for the given tmin, tmax
    tmin_samples = int(raw.info['sfreq'] * reject_tmin)
    tmax_samples = int(raw.info['sfreq'] * reject_tmax)
    n_samples_in_epoch = tmax_samples - tmin_samples

    # get indices of epochs
    starts = (events[:, 0] + tmin_samples).astype(np.int)
    stops = (events[:, 0] + tmax_samples).astype(np.int)
    epoch_indices = vrange(starts, stops)
    # epoch data
    data = raw.get_data()
    epoched_data = data[:, epoch_indices].reshape(
        (data.shape[0], n_samples_in_epoch, events.shape[0]), order='F')
    # check rejection criteria
    max_voltage = np.max(epoched_data, axis=1)
    min_voltage = np.min(epoched_data, axis=1)
    reject = abs(max_voltage - min_voltage) > threshold
    reject = np.sum(reject, axis=0) > 1
    # add 'bad' annotation for the entire epoch duration (tmin to tmax) of
    # epochs that contained peak to peak voltage above the threshold
    raw._annotations.append(starts[reject], reject_tmax - reject_tmin, 'bad')
