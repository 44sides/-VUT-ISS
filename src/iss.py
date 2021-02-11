import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import lfilter

length_s = 16000  # 1 second

# selected 1 second part for maskoff (45000:61000)
s_off, fs = sf.read('audio/maskoff_tone.wav')
s_off = s_off[45000:61000]
s_off -= np.mean(s_off)  # centralization

# selected 1 second part for maskon (45500:61500)
s_on, _ = sf.read('audio/maskon_tone.wav')
s_on = s_on[45500:61500]
s_on -= np.mean(s_on)  # centralization

sentence_off, _ = sf.read('audio/maskoff_sentence.wav')  # 57003 samples
sentence_on, _ = sf.read('audio/maskon_sentence.wav')  # 49493 samples

frame_sec = 0.02
frame_s = int(frame_sec * fs)
overlap_sec = 0.01
overlap_s = int(fs * overlap_sec)

miniFrames_number = int(length_s / overlap_s)
frames_number = int((length_s / overlap_s) - 1)

# Splitting into frames 2D matrix. (3)
# Rows equal to the number of frames. # Columns equal to the frame sample size or the length of each DFT
frames_off = np.ndarray((frames_number, frame_s))
frames_on = np.ndarray((frames_number, frame_s))
s_index_from = 0
s_index_to = frame_s
for i in range(0, frames_number):
    frames_off[i] = s_off[s_index_from:s_index_to]
    frames_on[i] = s_on[s_index_from:s_index_to]
    s_index_from += overlap_s
    s_index_to += overlap_s

# random frames drawing
_, ax = plt.subplots(2, 1, figsize=(9, 3))
ax[0].plot(np.arange(frame_s) / fs, frames_off[77])
ax[0].set_xlabel('$time$')
ax[0].set_title('Mask off frame 77')
ax[1].plot(np.arange(frame_s) / fs, frames_on[77])
ax[1].set_xlabel('$time$')
ax[1].set_title('Mask on frame 77')
plt.tight_layout()
# plt.savefig('img/SMALL_frames.png')
plt.show()

_, ax = plt.subplots(2, 1, figsize=(9, 3))
# frame drawing
ax[0].plot(np.arange(frame_s)/fs, frames_off[25])
ax[0].set_xlabel('$time$')
ax[0].set_title('Mask off Frame 25')
plt.grid(alpha=0.5, linestyle='--')


# Center Clipping (4)
clipping_frames_off = np.ndarray((frames_number, frame_s))
clipping_frames_on = np.ndarray((frames_number, frame_s))

for i in range(0, frames_number):
    AbsMax_value_off = abs(max(frames_off[i], key=abs))
    negAbsMax_value_off = -1 * AbsMax_value_off
    AbsMax_value_on = abs(max(frames_on[i], key=abs))
    negAbsMax_value_on = -1 * AbsMax_value_on

    for y in range(0, frame_s):
        if frames_off[i][y] > 0.7 * AbsMax_value_off:
            clipping_frames_off[i][y] = 1
        elif frames_off[i][y] < 0.7 * negAbsMax_value_off:
            clipping_frames_off[i][y] = -1
        else:
            clipping_frames_off[i][y] = 0

    for y in range(0, frame_s):
        if frames_on[i][y] > 0.7 * AbsMax_value_on:
            clipping_frames_on[i][y] = 1
        elif frames_on[i][y] < 0.7 * negAbsMax_value_on:
            clipping_frames_on[i][y] = -1
        else:
            clipping_frames_on[i][y] = 0

# clipping drawing
ax[1].plot(np.arange(frame_s) / fs, clipping_frames_off[25])
ax[1].set_xlabel('$time$')
ax[1].set_title('Mask off Clipping 25')
plt.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
# plt.savefig('img/SMALL_mask_off_frame.png')
plt.show()

# Auto Correlation
Rxx_off = np.zeros((frames_number, frame_s))
Rxx_on = np.zeros((frames_number, frame_s))

for i in range(0, frames_number):
    for k in range(0, frame_s):
        for n in range(0, frame_s):
            if 0 <= (n - k) < frame_s:
                Rxx_off[i][k] += clipping_frames_off[i][n] * clipping_frames_off[i][n - k]
                Rxx_on[i][k] += clipping_frames_on[i][n] * clipping_frames_on[i][n - k]

# Searching of frames fundamental frequency
f0_off = np.ndarray(frames_number)
f0_on = np.ndarray(frames_number)

for i in range(0, frames_number):
    Rxx_limit_off = Rxx_off[i][32:]  # 32 s = 500 Hz threshold
    Rxx_limit_on = Rxx_on[i][32:]  # 32 s = 500 Hz threshold
    lag_index_off = np.argmax(Rxx_limit_off) + 32
    lag_index_on = np.argmax(Rxx_limit_on) + 32
    f0_off[i] = fs / lag_index_off
    f0_on[i] = fs / lag_index_on

# Mean and dispersion calculation
f0_off_mean = np.mean(f0_off)
f0_on_mean = np.mean(f0_on)
f0_off_dispersion = np.std(f0_off) ** 2
f0_on_dispersion = np.std(f0_on) ** 2

print("Means of fundamental frequencies:", f0_off_mean, "- mask off;", f0_on_mean, "- mask on")
print("Dispersions of fundamental frequencies:", f0_off_dispersion, "- mask off;", f0_on_dispersion, "- mask on")


# correlation drawing
_, ax = plt.subplots(2, 1, figsize=(9, 3))

Rxx_limit_draw = Rxx_off[25][32:]  # 32 s = 500 Hz threshold
lag_index_draw = np.argmax(Rxx_limit_draw) + 32

ax[0].plot(np.arange(frame_s), Rxx_off[25])
ax[0].set_xlabel('$sample$')
ax[0].set_title('Mask off Auto Correlation 25')
ax[0].stem([lag_index_draw], [Rxx_off[25][lag_index_draw]], linefmt='r-', label='lag')
ax[0].axvline(x=32, color='black', label='threshold')
ax[0].legend(loc='best')
plt.grid(alpha=0.5, linestyle='--')

# fundamental frequency drawing
ax[1].plot(np.arange(frames_number), f0_off, label='mask off')
ax[1].plot(np.arange(frames_number), f0_on, label='mask on')
ax[1].set_xlabel('$frame$')
ax[1].set_title('Fundamental frequency of frames')
ax[1].legend(loc='best')
plt.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
# plt.savefig('img/SMALL_correlation_fundFrequency.png')
plt.show()


# Spectrogram (5)

def DFT(frame):
    frame = np.asarray(frame, dtype=complex)
    size = frame.size
    dft = np.ndarray(size, dtype=complex)

    for q in range(size):
        s = 0
        for j in range(size):
            s += frame[j] * np.exp(-2j * np.pi * j * q * np.divide(1, size, dtype=complex))
        dft[q] = s

    return dft


N = 1024

# DFT
frames_dft_off = np.zeros((frames_number, N), dtype=complex)
frames_dft_on = np.zeros((frames_number, N), dtype=complex)

frames_new_off = np.zeros((frames_number, N))
frames_new_on = np.zeros((frames_number, N))
for i in range(frames_number):
    for y in range(frame_s):
        frames_new_off[i][y] = frames_off[i][y]
        frames_new_on[i][y] = frames_on[i][y]

for i in range(frames_number):
    frames_dft_off[i] = DFT(frames_new_off[i])
    frames_dft_on[i] = DFT(frames_new_on[i])

# PSD
Pk_off = np.ndarray((frames_number, N))
Pk_on = np.ndarray((frames_number, N))
for i in range(frames_number):
    Pk_off[i] = 10 * np.log10(np.abs(frames_dft_off[i]) ** 2)
    Pk_on[i] = 10 * np.log10(np.abs(frames_dft_on[i]) ** 2)

# avoidance of symmetry
Pk_half_off = np.ndarray((frames_number, N // 2 + 1))
Pk_half_on = np.ndarray((frames_number, N // 2 + 1))
for i in range(frames_number):
    Pk_half_off[i] = Pk_off[i][:N // 2 + 1]
    Pk_half_on[i] = Pk_on[i][:N // 2 + 1]

# spectrogram drawing
_, ax = plt.subplots(2, 1, figsize=(6, 3))

rel = ax[0].imshow(np.transpose(Pk_half_off), extent=[0, 1, 0, 8000], aspect='auto', origin='lower')
ax[0].set_xlabel('$time[s]$')
ax[0].set_ylabel('$Frequency[Hz]$')
ax[0].set_title('Spectrogram Mask off')
bar_db = plt.colorbar(rel, ax=ax[0])
bar_db.set_label('$PSD[dB]$', rotation=270, labelpad=15)

rel = ax[1].imshow(np.transpose(Pk_half_on), extent=[0, 1, 0, 8000], aspect='auto', origin='lower')
ax[1].set_xlabel('$time[s]$')
ax[1].set_ylabel('$Frequency[Hz]$')
ax[1].set_title('Spectrogram Mask on')
bar_db = plt.colorbar(rel, ax=ax[1])
bar_db.set_label('$PSD[dB]$', rotation=270, labelpad=15)

plt.tight_layout()
# plt.savefig('img/SMALL_spectrogram.png')
plt.show()


# Frequency response (6)
frames_freqResp_H = np.ndarray((frames_number, N), dtype=complex)
for i in range(frames_number):
    frames_freqResp_H[i] = frames_dft_on[i] / frames_dft_off[i]

frames_freqResp_H_t = np.absolute(np.array(frames_freqResp_H).T)

freqResponse = np.ndarray(N)
for i in range(N):
    freqResponse[i] = np.mean(frames_freqResp_H_t[i])   

freqResponse_half = freqResponse[:N // 2 + 1]

freqResponse_log = 10 * np.log10(np.abs(freqResponse_half) ** 2)

# frequency response drawing
plt.figure(figsize=(6, 3))
plt.plot(np.arange(N//2 + 1), freqResponse_log)
plt.gca().set_title('Frequency response')
plt.tight_layout()
# plt.savefig('img/SMALL_freq_response.png')
plt.show()


# IDFT (7)

def IDFT(dft):
    dft = np.asarray(dft, dtype=complex)
    size = dft.size
    idft = np.ndarray(size, dtype=complex)

    for q in range(size):
        s = 0
        for j in range(size):
            s += dft[j] * np.exp(2j * np.pi * q * j * np.divide(1, size, dtype=complex))
        idft[q] = np.divide(s, size, dtype=complex)

    return idft


impulseResponse = IDFT(freqResponse).real
impulseResponse = impulseResponse[:N // 2 + 1]

# impulse response drawing
plt.figure(figsize=(6, 3))
plt.plot(np.arange(N // 2 + 1), impulseResponse)
plt.gca().set_title('Impulse response')
plt.tight_layout()
# plt.savefig('img/SMALL_impulse_response.png')
plt.show()


# Simulation (8)
sentence_off = sentence_off[9050:45050]  # empty start cutting
sentence_off -= np.mean(sentence_off)  # centralization
sentence_on = sentence_on[13000:49000]  # empty start cutting
sentence_on -= np.mean(sentence_on)  # centralization

simulation_sentence_mask_on = lfilter(impulseResponse, [1], sentence_off)
sf.write('audio/sim_maskon_sentence.wav', simulation_sentence_mask_on, fs)

s_off, fs = sf.read('audio/maskoff_tone.wav')
simulation_tone_mask_on = lfilter(impulseResponse, [1], s_off)
sf.write('audio/sim_maskon_tone.wav', simulation_tone_mask_on, fs)

# simulation drawing
_, ax = plt.subplots(3, 1, figsize=(9, 3))

ax[0].plot(np.arange(36000), sentence_off, label='mask off sntnc')
ax[0].plot(np.arange(36000), sentence_on, label='mask on sntnc')
ax[0].legend(loc='best')

ax[1].plot(np.arange(36000), sentence_off, label='mask off sntnc')
ax[1].plot(np.arange(36000), simulation_sentence_mask_on, label='mask on sntnc sim')
ax[1].legend(loc='best')

ax[2].plot(np.arange(36000), sentence_on, label='mask on sntnc')
ax[2].plot(np.arange(36000), simulation_sentence_mask_on, label='mask on sntnc sim')
ax[2].legend(loc='best')

plt.tight_layout()
# plt.savefig('img/SMALL_simulation.png')
plt.show()
