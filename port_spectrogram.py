import torch
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import torchlibrosa as tl
from nnAudio import features

import matplotlib.pyplot as plt
plt.ion()

# Port mel-scaled spectrogram code to Torch

sample_rate = 44100
duration = 6.0
num_samples = int(sample_rate*duration)
n_mels = 512
hop = 500
n_fft = 2*2048
target_shape = (1, 512, 512)

def plot_comp(data, labels, y_scale='log'):    
    num_methods = len(data)
    N, H, W = data[0].shape
    
    fig, ax = plt.subplots(nrows=N, ncols=num_methods, sharex=True)
    fig.tight_layout()

    for i_clip in range(N):
        for i_method in range(num_methods):
            img = data[i_method][i_clip]
            db = librosa.amplitude_to_db(img.numpy().T, ref=np.max)
            axis = ax[i_clip][i_method]
            axis.set(title=labels[i_method])
            librosa.display.specshow(db, sr=sample_rate, y_axis=y_scale, x_axis='time', ax=axis)
    
    return fig, ax

def f_tf(waveform):
    z = tf.signal.stft(waveform, frame_length=2*2048, frame_step=hop, pad_end=True)
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
            tl.Spectrogram(n_fft=n_fft, hop_length=hop, power=1),
            tl.LogmelFilterBank(n_fft=n_fft, sr=sample_rate, n_mels=n_mels, fmin=0.0, fmax=0.5*sample_rate, is_log=False)
        )

    melspectrogram = _converter(waveform)
    return torch.log1p(melspectrogram)[:, 0, :target_shape[1], :target_shape[2]]

def comp(waveform, wftf, converter=None):
    z1 = tf.signal.stft(wftf, frame_length=n_fft, frame_step=hop, pad_end=True)
    z2 = tl.Spectrogram(n_fft=n_fft, hop_length=hop, power=1)(waveform)[:, 0, :, :]
    z3 = features.STFT(n_fft=n_fft, hop_length=hop, fmin=0, fmax=0.5*sample_rate, sr=sample_rate, output_format='Magnitude')(waveform).permute(0, 2, 1)

    mag1 = tf.abs(z1)
    mag2 = torch.abs(z2)
    mag3 = torch.abs(z3)

    fig, ax = plot_comp([mag1, mag2, mag3], ['tf.signal.stft', 'tl.Spectrogram', 'nnAudio.STFT'], y_scale='log')
    fig.set_size_inches(30/2.54, 18/2.54)
    plt.savefig('stft_comp.png')
    plt.close('all')

    filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=mag1.shape[-1],
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=0.5*sample_rate)
    
    ms1 = tf.tensordot(mag1, filterbank, 1)
    ms2 = tl.LogmelFilterBank(n_fft=n_fft, sr=sample_rate, n_mels=n_mels, fmax=0.5*sample_rate, is_log=False)(mag2)
    ms3 = features.MelSpectrogram(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, power=1, fmin=0, fmax=0.5*sample_rate, hop_length=hop)(waveform).permute(0, 2, 1)
    
    #if converter is None:
    #    converter = torch.nn.Sequential(
    #        tl.Spectrogram(n_fft=n_fft, hop_length=hop, power=1),
    #        tl.LogmelFilterBank(n_fft=n_fft, sr=sample_rate, n_mels=n_mels, fmin=0.0, fmax=0.5*sample_rate, is_log=False)
    #    )
    #ms2 = converter(waveform.unsqueeze(0)) 

    s1_scaled = ms1 / tf.reduce_max(ms1)
    s2_scaled = ms2 / ms2.max()
    s3_scaled = ms3 / ms3.max()
    
    fig, ax = plot_comp([s1_scaled, s2_scaled, s3_scaled], ['tensorflow', 'torchlibrosa', 'nnAudio'], y_scale='mel')
    fig.set_size_inches(30/2.54, 18/2.54)
    plt.savefig('mel_comp.png')
    plt.close('all')

waveform, sr = librosa.load('C:/Users/Erik/code/timbrer/data/wav/shakuhachi.wav', sr=sample_rate, duration=2*duration)
assert sr == sample_rate

# Add batch dim
waveform = np.stack([waveform[:num_samples], waveform[num_samples:]], axis=0)

wfpt = torch.from_numpy(waveform).to('cpu')
wftf = tf.constant(waveform)

# DEBUG
comp(wfpt, wftf)

print('Done')