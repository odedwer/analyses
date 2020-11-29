# %%
# ##import
import math
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import os.path as op
import seaborn as sns
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.time_frequency import tfr_morlet, tfr_multitaper, psd_multitaper, psd_welch, tfr_stockwell
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy import signal
from preprocess_utilities import *
import gc
###starts with an epoch object after artifact removal
###all further analysis will happen here


# %% parameters
stimuli = ['short_scrambled', 'long_scrambled','short_face', 'long_face',
           'short_obj', 'long_obj','short_body', 'long_body']
stimuli=["short_word","long_word"]
#chosen_s_trigs = stimuli[6]
#chosen_l_trigs = stimuli[7]

freq_range = [5, 200]
base_correction = (-0.25, -.10)  # when epoch starts at -0.400
correction_mode = 'logratio'
# alpha = [8, 12]
# beta = [13, 30]
# narrowgamma = [31, 60]
# high_gamma = [60, 150]
freqsH = np.logspace(5, 7.6, 50, base=2)
freqs_all = np.logspace(1, 7.6, 40, base=2)
freqsL = np.logspace(1, 5, 8, base=2)
n_perm = 500
save_path = 'SavedResults/S4'
# %% # triggers for ERP - choose triggers, create topomap and ERP graph per chosen electrode ##n all
curr_epochs = epochs_filt  # choose the triggers you want to process
evoked = curr_epochs.average()
evoked.plot_topomap(outlines='skirt', times=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9,1.1,1.3,1.5,1.7])

# %% # visualize ERP by electrode
filt_epochs_plot = curr_epochs["short_face","long_face"].plot_image(picks=['B30'])
#%% calculate power and save
power_long_H = tfr_morlet(epochs[stimuli[1]], freqs=freqsH, average=False,
                          n_cycles=freqsH / 10, use_fft=True,
                          return_itc=False, decim=3, n_jobs=12)
power_long_H=power_long_H.average()
power_long_H.save(os.path.join(save_path, 'S4_audWord_avgRef_long_trials_high_freq-pow.fif'))
gc.collect()
#%%
power_short_H = tfr_morlet(epochs[stimuli[0]], freqs=freqsH, average=False,
                           n_cycles=freqsH / 10, use_fft=True,
                           return_itc=False, decim=3, n_jobs=12)
power_short_H = power_short_H.average()
power_short_H.save(os.path.join(save_path, 'S4_audWord_avgRef_short_trials_high_freq-pow.fif'))
gc.collect()
#%% low freqs tfr
power_long_L = tfr_morlet(epochs['long_anim', 'long_obj', 'long_face', 'long_body'], freqs=freqsL, average=False,
                          n_cycles=freqsL / 10, use_fft=True,
                          return_itc=False, decim=3, n_jobs=5)
power_long_L.save(os.path.join(save_path, 'S3_vis_detrended_ord10_10s_windows_long_trials_low_freq-pow.fif'))
del power_long_L; gc.collect()
power_short_L = tfr_morlet(epochs['short_anim', 'short_obj', 'short_face', 'short_body'], freqs=freqsL, average=False,
                           n_cycles=freqsL / 10, use_fft=True,
                           return_itc=False, decim=3, n_jobs=5)
power_short_L.save(os.path.join(save_path, 'S3_vis_detrended_ord10_10s_windows_short_trials_low_freq-pow.fif'))
del power_short_L
gc.collect()
# %% # time frequency analysis - high freqs tfrs

# %%
power = mne.time_frequency.read_tfrs("SavedResults/S3/S3_vis_detrended_ord10_10s_windows_short_trials_high_freq-pow.fif")
power=power[0].average() # get onlyaverage induced response
power.plot_topo(baseline=base_correction, mode=correction_mode, title='Average power',
                       tmin=-0.3, tmax=1.8,vmin=-.25, vmax=.25, layout_scale=.5 )

# %% show ERP after hilbert
## filter raw data using iir butterwoth filter of order 3
raw_hilb = []
nbands = 4  # starting from 50, jumping by 20
for i in range(nbands):
    curr_l_freq = (i + 1) * 20 + 30
    curr_h_freq = curr_l_freq + 20
    raw_bp = raw.copy().filter(l_freq=curr_l_freq, h_freq=curr_h_freq, method='iir',
                               iir_params=dict(order=3, ftype='butter'))
    raw_hilb.append(raw_bp.apply_hilbert(envelope=True))  # compute envelope of analytic signal
    raw_hilb[i]._data[:] = 10 * np.log10(raw_hilb[i]._data[:] ** 2)  # change to dB
    raw_hilb[i]._data = raw_hilb[i]._data[:] - np.reshape(raw_hilb[i]._data[:].mean(1),(raw_hilb[i].info['nchan'],1))
    # demean to apply correction
    print("finished band of " + str(curr_l_freq) + " - " + str(curr_h_freq))
    input()

##  compute mean of all filter bands
raw_hilb[0]._data = (raw_hilb[0]._data + raw_hilb[1]._data + raw_hilb[2]._data + raw_hilb[3]._data) / nbands
raw_hilb[0].filter(l_freq=1,h_freq=30) # should I?
epochs_hilb = mne.Epochs(raw_hilb[0], events, event_id=event_dict_vis,
                         tmin=-0.4, tmax=1.9, baseline=None,
                         reject_tmin=-.1, reject_tmax=1.5,  # reject based on 100 ms before trial onset and 1500 after
                         preload=True, reject_by_annotation=True)
#del raw_hilb
# apply hilbert
epochs_hilb.apply_baseline((-.3, -.05), verbose=True)


# %% show ERP after hilbert
check_electrode = "A11"
#epochs_hilb['short_word'].plot_image(picks=[check_electrode],title=check_electrode+"short HFB average power")
epochs_hilb['long_face'].plot_image(picks=[check_electrode],title=check_electrode+" long HFB average power")

#%%
evokedHFB_L = epochs_hilb["long_face"].average()
evokedHFB_S = epochs_hilb["short_face"].average()
mne.viz.plot_compare_evokeds({"Long":evokedHFB_L,"Short":evokedHFB_S},check_electrode,title="Long and short HFB",vlines=[.8,1.5])

#%%
short_epo_hilb = epochs_hilb["short_word"].average()
long_epo_hilb = epochs_hilb["long_word"].average()
long_m_short_hilb = mne.combine_evoked([long_epo_hilb,short_epo_hilb],[1,-1])
plt.plot(raw.ch_names[:256],np.sum((long_m_short_hilb.data[:,614:970]),axis=1))
plt.plot(raw.ch_names[:256],np.sum((evokedHFB_L.data[:,300:970]),axis=1))
long_m_short_hilb.plot_topo()
# %%

# %% compute point by point long-short t-test and then duration tracking score, for each electrode
# for our cause, the duration tracking period is end of first to end of second

dt_scores = duration_tracking(epochs_hilb[stimuli[1:6:2]],epochs_hilb[stimuli[0:6:2]],time_diff=[0.8,1.5])
mne.viz.plot_topomap(dt_scores,epochs_hilb.info)
plt.bar(raw.ch_names[0:256],dt_scores)
plt.plot(raw.ch_names[0:256],dt_scores)


# %%
# power_list = []
# # widths = [0.2, 0.4]
# widths = [0.2]
# for width in widths:
#     power_list.append(tfr_morlet(ds_epochs[["long_word"]], freqs=freqs, use_fft=True,
#                                  n_cycles=np.concatenate([3 * np.ones(9), 12 * np.ones(29)]), verbose=True, average=False, return_itc=False))
# # n_cycles = freqs * width
#
# # %%
# power_list_baseline_corrected = []
# for i in range(len(power_list)):
#     power_list_baseline_corrected.append(power_list[i].copy().apply_baseline(mode='logratio', baseline=(-.200, 0)))
# # %%
# power_avg = [power.average() for power in power_list_baseline_corrected]
# # %%
# power_avg[0].plot_topo()
#
# # %%
# for i in range(len(power_list)):
#     power_list[i].crop(0., 1.6)
#
# evoked = ds_epochs.average()
# evoked.crop(0., 1.6)
# times = evoked.times
#
# del evoked
#
# # %%
# for j in range(18, 145):
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
#     for power, ax, width in zip(power_list, axs, widths):
#         epochs_power = power.data[:, j, :, :]  # take the 1 channel
#         threshold = 2.5
#         T_obs, clusters, cluster_p_values, H0 = \
#             mne.stats.permutation_cluster_1samp_test(epochs_power, n_permutations=n_perm, tail=0)
#         T_obs_plot = np.nan * np.ones_like(T_obs)
#         for c, p_val in zip(clusters, cluster_p_values):
#             if p_val <= 0.05:
#                 T_obs_plot[c] = T_obs[c]
#         vmax = np.max(np.abs(T_obs))
#         vmin = -vmax
#         ax.imshow(T_obs, cmap=plt.cm.RdBu_r,
#                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                   aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#         ax.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
#                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
#                   aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
#         ax.set_xlabel('Time (ms)')
#         ax.set_ylabel('Frequency (Hz)')
#         # ax.title('Induced power (%s)' % j)
#         # power.plot([j], baseline=(0., 0.2), mode='logratio', axes=ax, colorbar=True if width == widths[-1] else False,
#         #            show=False, vmin=-.6, vmax=.6)
#         ax.set_title('Sim: Using multitaper, width = {:0.1f}'.format(width))
#     plt.show()
#     if input():
#         break
#
# # %%
# for j in range(197, 200):
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
#     for power, ax, width in zip(power_list, axs, widths):
#         avg = power.average()
#         vmax = np.max(np.abs(avg.data))
#         vmin = -vmax
#         avg.plot([j], axes=ax, colorbar=True if width == widths[-1] else False,
#                  show=False, vmin=vmin, vmax=vmax)
#         ax.set_title('Sim: Using multitaper, width = {:0.1f}'.format(width))
#     plt.show()
#     if input():
#         break
#
# # %% # running with low frequencies to ensure we see ERP
# power_low = tfr_morlet(epochs, freqs=[3, 6, 8, 10, 12, 14, 16], average=False,
#                         n_cycles=np.concatenate([3 * np.ones(7)]), use_fft=True,
#                         return_itc=False, decim=3, n_jobs=1)
# power_low = power_low.average()
