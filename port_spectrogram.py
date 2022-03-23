import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt

import torchlibrosa as tl

from torchaudio.transforms import Spectrogram, MelSpectrogram

# Port mel-scaled spectrogram code to Torch

sample_rate = 44100
duration = 6.0
num_samples = int(sample_rate*duration)
n_mels = 512
target_shape = (1, 512, 512)

def f_tf(waveform):
    z = tf.signal.stft(waveform, frame_length=2*2048, frame_step=500, pad_end=True)
    magnitudes = tf.abs(z)
    filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels, #80
        num_spectrogram_bins=magnitudes.shape[-1],
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=0.5*sample_rate) #8k
    melspectrogram = tf.tensordot(magnitudes, filterbank, 1)
    return tf.math.log1p(melspectrogram)[:, :target_shape[1], :target_shape[2]]

_converter = None
def f_pt(waveform):
    global _converter

    if _converter is None:
        _converter = torch.nn.Sequential(
            tl.Spectrogram(n_fft=2*2048, hop_length=500, power=1),
            tl.LogmelFilterBank(n_fft=2*2048, sr=sample_rate, n_mels=n_mels, fmin=0.0, fmax=0.5*sample_rate, is_log=False)
        )

    melspectrogram = _converter(waveform)
    return torch.log1p(melspectrogram)[:, 0, :target_shape[1], :target_shape[2]]

def comp(waveform, wftf, converter=None):
    z1 = tf.signal.stft(wftf, frame_length=2*2048, frame_step=500, pad_end=True)
    #z = tl.Spectrogram(n_fft=2*2048, hop_length=500, power=1)(waveform.unsqueeze(0))

    # TODO: add https://github.com/KinWaiCheuk/nnAudio?

    magnitudes1 = tf.abs(z1)

    #combined = np.hstack([magnitudes1.numpy().T, magnitudes[0, 0, :H, :W].cpu().numpy().T])
    #db = librosa.amplitude_to_db(combined, ref=np.max)
    #librosa.display.specshow(db, sr=sample_rate, y_axis='log', x_axis='time')

    filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=magnitudes1.shape[-1],
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=0.5*sample_rate)
    
    melspectrogram1 = tf.tensordot(magnitudes1, filterbank, 1)
    #melspectrogram = tl.LogmelFilterBank(n_fft=2*2048, sr=sample_rate, n_mels=n_mels, fmax=0.5*sample_rate, is_log=False)(magnitudes)
    
    if converter is None:
        converter = torch.nn.Sequential(
            tl.Spectrogram(n_fft=2*2048, hop_length=500, power=1),
            tl.LogmelFilterBank(n_fft=2*2048, sr=sample_rate, n_mels=n_mels, fmin=0.0, fmax=0.5*sample_rate, is_log=False)
        )

    melspectrogram = converter(waveform.unsqueeze(0))

    s1_scaled = (melspectrogram1 / tf.reduce_max(melspectrogram1)).numpy()
    s2_scaled = (melspectrogram / melspectrogram.max())[0, 0, :, :].cpu().numpy()
    combined = np.hstack([s1_scaled.T, s2_scaled.T])

    plt.figure(figsize=(15, 15))
    librosa.display.specshow(librosa.amplitude_to_db(combined, ref=np.max), sr=sample_rate, y_axis='mel', x_axis='time')
    plt.show()

    return tf.math.log1p(melspectrogram)[:, :target_shape[1], :target_shape[2]]


waveform, sr = librosa.load('C:/Users/Erik/code/timbrer/data/wav/shakuhachi.wav', sr=sample_rate, duration=2*duration)
assert sr == sample_rate

# Add batch dim
waveform = np.stack([waveform[:num_samples], waveform[num_samples:]], axis=0)

wfpt = torch.from_numpy(waveform).to('cpu')
wftf = tf.constant(waveform)

v1, v2 = f_pt(wfpt).numpy().transpose((0, 2, 1))
v3, v4 = f_tf(wftf).numpy().transpose((0, 2, 1))
comb1 = np.hstack([v1 / v1.max(), v3 / v3.max()])
comb2 = np.hstack([v2 / v2.max(), v4 / v4.max()])

plt.figure(figsize=(15, 25))
librosa.display.specshow(librosa.amplitude_to_db(comb2, ref=np.max), sr=sample_rate, y_axis='mel', x_axis='time')
plt.show()

print('Done')