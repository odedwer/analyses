import numpy as np
from scipy.io.wavfile import write
import scipy
import matplotlib.pyplot as plt


def db2mag(db):
    return 10 ** (db / 10)


def gen_sound(freq, amplitude, duration=1., fs=44100):
    return amplitude * (np.sin(2 * np.pi * np.arange(fs * duration) * freq / fs)).astype(np.float32)


# %%
freqs = np.round(np.logspace(start=np.log10(250), stop=np.log10(2000),
                             num=180))  # base frequencies - logspace 250-2000 (198.46e**0.231x)
harmonic_db = [-3, -6, -9, -12]  # number of db here will determine the number of harmonics
amp = 1  # amplitude of base frequency
dur = .800  # seconds
ramp_percentage = 0.01/dur  # how much of the sound is being ramped
num_of_harmonics = len(harmonic_db)
db_of_harmonics = [amp * db2mag(db) for db in harmonic_db]
fs = 44100
results = []
for f in freqs:
    samples = gen_sound(f, amp, dur)
    num_of_ramp_samples = int(samples.size * ramp_percentage)
    for i in range(2, num_of_harmonics + 1):  # generate the harmonics
        print(f"curr freq is {i*f} in db {db_of_harmonics[i-2]}",end=", ")
        samples += gen_sound(i * f, db_of_harmonics[i-2], dur, fs)
    print()
    # create the ramps
    up_ramp = np.linspace(0, 1, num_of_ramp_samples)
    down_ramp = np.linspace(1, 0, num_of_ramp_samples)
    samples[:num_of_ramp_samples] *= up_ramp
    samples[samples.size - num_of_ramp_samples:] *= down_ramp
    samples /= np.max(np.abs(samples))
    results.append(samples)
    write("base_freq_%d_with_%d_harmonics_800ms.wav" % (f, num_of_harmonics), fs, samples)
