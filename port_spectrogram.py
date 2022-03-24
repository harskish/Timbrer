import wave
import torch
import torch.optim as optim
import torch.nn.functional as F
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
hop = 517 # produces time res 512 (TODO: divisibility & padding?)
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

def play_audio(waveform):
    if torch.is_tensor(waveform):
        waveform = waveform.detach().cpu()
    
    if not isinstance(waveform, np.ndarray):
        waveform = waveform.numpy()
    
    mag = np.abs(waveform).max()
    if mag > 1.0:
        waveform /= mag

    import sounddevice as sd
    sd.play(waveform, sample_rate, blocking=True)

cache = {}

def stft_tf(waveform):
    z = tf.signal.stft(waveform, frame_length=n_fft, frame_step=hop, pad_end=True)
    return tf.abs(z)

def mel_tf(waveform):
    magnitudes = stft_tf(waveform)
    filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels, #80
        num_spectrogram_bins=magnitudes.shape[-1],
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=0.5*sample_rate) #8k
    melspectrogram = tf.tensordot(magnitudes, filterbank, 1)
    return tf.math.log1p(melspectrogram)

def stft_pt_tl(waveform):
    if 'tl_stft' not in cache:
        cache['tl_stft'] = tl.Spectrogram(n_fft=n_fft, hop_length=hop, power=1).cuda()
    
    return cache['tl_stft'].to(waveform.device)(waveform)[:, 0, :, :]

def mel_pt_tl(waveform):
    if 'tl_mel' not in cache:
        cache['tl_mel'] = torch.nn.Sequential(
            tl.Spectrogram(n_fft=n_fft, hop_length=hop, power=1),
            tl.LogmelFilterBank(n_fft=n_fft, sr=sample_rate, n_mels=n_mels, fmin=0.0, fmax=0.5*sample_rate, is_log=False)
        ).cuda()

    melspectrogram = cache['tl_mel'].to(waveform.device)(waveform)
    return torch.log1p(melspectrogram)[:, 0, :, :]

def stft_pt_nn(waveform):
    if 'nn_stft' not in cache:
        cache['nn_stft'] = features.STFT(n_fft=n_fft, hop_length=hop, fmin=0, fmax=0.5*sample_rate, sr=sample_rate, output_format='Magnitude').cuda()
    
    return cache['nn_stft'].to(waveform.device)(waveform).permute(0, 2, 1)

def mel_pt_nn(waveform):
    if 'nn_mel' not in cache:
        cache['nn_mel'] = features.MelSpectrogram(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, power=1, fmin=0, fmax=0.5*sample_rate, hop_length=hop).cuda()

    melspectrogram = cache['nn_mel'].to(waveform.device)(waveform).permute(0, 2, 1)
    return torch.log1p(melspectrogram)

def find_bin_size(n_octaves, n_bins, fmin, sr):
    max_octaves = np.log2(0.5*sr/fmin)
    assert n_octaves < max_octaves, 'n_octaves too large for range [fmin,sr/2]'
    
    bins_per_octave = 0
    fmax_t = float('inf')

    while fmax_t > sr / 2:
        bins_per_octave += 1
        
        # Calculate the lowest frequency bin for the top octave kernel
        fmin_t = fmin * 2 ** (n_octaves - 1)
        remainder = n_bins % bins_per_octave

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = fmin_t * 2 ** ((remainder - 1) / bins_per_octave)

    return bins_per_octave

def cqt_pt_nn(waveform):
    if 'nn_cqt' not in cache:
        fmin = 32.7
        bins = 1024 #512
        bin_size = 109 #find_bin_size(n_octaves=12, n_bins=bins, fmin=fmin, sr=sample_rate)
        cache['nn_cqt'] = features.CQT2010v2(
            sr=sample_rate, n_bins=bins, bins_per_octave=bin_size, fmin=fmin, fmax=0.5*sample_rate, hop_length=512).cuda()

    cqt = cache['nn_cqt'].to(waveform.device)(waveform).permute(0, 2, 1)
    return cqt

def comp_fwd(waveform, wftf):
    # Spectrogram comparison
    mag1 = stft_tf(wftf)
    mag2 = stft_pt_tl(waveform)
    mag3 = stft_pt_nn(waveform)

    fig, ax = plot_comp([mag1, mag2, mag3], ['tf.signal.stft', 'tl.Spectrogram', 'nnAudio.STFT'], y_scale='log')
    fig.set_size_inches(30/2.54, 18/2.54)
    plt.savefig('stft_comp.png')
    plt.close('all')
    
    # Mel-scaled spectrogram comparison
    ms1 = mel_tf(wftf)
    ms2 = mel_pt_tl(waveform)
    ms3 = mel_pt_nn(waveform)

    s1_scaled = ms1 / tf.reduce_max(ms1)
    s2_scaled = ms2 / ms2.max()
    s3_scaled = ms3 / ms3.max()
    
    fig, ax = plot_comp([s1_scaled, s2_scaled, s3_scaled], ['tensorflow', 'torchlibrosa', 'nnAudio'], y_scale='mel')
    fig.set_size_inches(30/2.54, 18/2.54)
    plt.savefig('mel_comp.png')
    plt.close('all')

def comp_bwd(waveform, wftf):
    # Griffin-Lim
    w1 = features.Griffin_Lim(n_fft, hop_length=hop, device='cuda')(stft_pt_nn(waveform).permute(0, 2, 1).detach().cuda())
    play_audio(torch.cat(w1.unbind(0)))
    
    # Optimization-based
    device = 'cuda'
    steps = 300
    func = cqt_pt_nn #cqt_pt_nn
    B = 1 if func == cqt_pt_nn else 2 # LBFGS scales poorly with batch size
    
    res = []
    for i in range(0, waveform.shape[0], B):
        wf = waveform[i:i+B].to(device)
        noise = torch.tensor(np.random.normal(scale=1e-1, size=wf.shape), dtype=torch.float32, device=device, requires_grad=True)
        opt = optim.LBFGS(params=[noise], lr=0.75, max_iter=steps, tolerance_change=0, tolerance_grad=0)
        target = func(wf).detach()
    
        iteration = 0
        def closure():
            nonlocal iteration
            iteration += 1

            opt.zero_grad()
            loss = F.mse_loss(func(noise), target)
            loss.backward()

            # TODO: waveforms not aligned somehow?
            waveform_loss = F.mse_loss(noise.detach(), wf)
            
            #if iteration == 4:
            #    play_audio(noise[0])

            # if iteration % 50 == 0:
            #     plot_comp([x.cpu().detach()], [f'iter{iteration}'], y_scale='mel')
            #     play_audio(noise[1])

            print('{}/{}: Img loss = {:.2e}, waveform loss: {:.2e}'.format(iteration, steps, loss.item(), waveform_loss.item()))
            return loss
        
        if isinstance(opt, optim.LBFGS):
            opt.step(closure)
        else:
            for _ in range(steps):
                opt.step(closure)
        
        for c in noise:
            res.append(c.detach())

    play_audio(torch.cat(res))
    print('Done')

if __name__ == '__main__':
    n_parts = 3
    #waveform, sr = librosa.load('C:/Users/Erik/code/timbrer/data/wav/shakuhachi.wav', sr=sample_rate, duration=n_parts*duration)
    waveform, sr = librosa.load('C:/Users/Erik/BATTLE BEAST - Eye of the Storm.wav', sr=sample_rate, offset=15, duration=n_parts*duration)
    assert sr == sample_rate

    # Round to integer number of parts
    n_parts = len(waveform) // num_samples
    waveform = waveform[:n_parts*num_samples]

    # Add batch dim
    waveform = np.stack([waveform[i*num_samples:(i+1)*num_samples] for i in range(n_parts)], axis=0)

    wfpt = torch.from_numpy(waveform).to('cpu')
    wftf = tf.constant(waveform)

    # Forward mode
    #comp_fwd(wfpt, wftf)

    # Reconstruction
    comp_bwd(wfpt, wftf)

    print('Done')