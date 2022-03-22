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
    z = tf.contrib.signal.stft(waveform, frame_length=2*2048, frame_step=500)
    magnitudes = tf.abs(z)
    filterbank = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels, #80
        num_spectrogram_bins=magnitudes.shape[-1].value,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=0.5*sample_rate) #8k
    melspectrogram = tf.tensordot(magnitudes, filterbank, 1)
    return tf.log1p(melspectrogram)[:, :target_shape[1], :target_shape[2]]

def f_pt(waveform, wftf):
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
    
    melspectrogram = torch.nn.Sequential(
        tl.Spectrogram(n_fft=2*2048, hop_length=500, power=1),
        tl.LogmelFilterBank(n_fft=2*2048, sr=sample_rate, n_mels=n_mels, fmin=0.0, fmax=0.5*sample_rate, is_log=False,
    ))(waveform.unsqueeze(0))

    plt.figure(figsize=(20,20))

    s1_scaled = (melspectrogram1 / tf.reduce_max(melspectrogram1)).numpy()
    s2_scaled = (melspectrogram / melspectrogram.max())[0, 0, :, :].cpu().numpy()

    combined = np.hstack([s1_scaled.T, s2_scaled.T])
    librosa.display.specshow(librosa.amplitude_to_db(combined, ref=np.max), sr=sample_rate, y_axis='mel', x_axis='time')
    plt.show()

    return tf.log1p(melspectrogram)[:, :target_shape[1], :target_shape[2]]


waveform, sr = librosa.load('C:/Users/Erik/code/timbrer/data/wav/shakuhachi.wav', sr=sample_rate, duration=duration)
assert sr == sample_rate

wfpt = torch.from_numpy(waveform).to('cpu')
wftf = tf.constant(waveform)

v = f_pt(wfpt, wftf)