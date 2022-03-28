import torch
import torch.optim as optim
import torch.nn.functional as F
#import tensorflow as tf
import numpy as np
from nnAudio import features
import audio_io
from pathlib import Path

import matplotlib.pyplot as plt
plt.ion()

# Port mel-scaled spectrogram code to Torch
from constants import *

def plot_comp(data, labels, y_scale='log'):
    import librosa
    import librosa.display

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

def _to_numpy(waveform):
    if torch.is_tensor(waveform):
        waveform = waveform.detach().cpu()
    if not isinstance(waveform, np.ndarray):
        waveform = waveform.numpy()
    return waveform

def play_audio(waveform, blocking=True):
    waveform = _to_numpy(waveform)
    mag = np.abs(waveform).max()
    if mag > 1.0:
        waveform /= mag

    import sounddevice as sd
    sd.play(waveform, sample_rate, blocking=blocking)

cache = {}

def stft_tf(waveform):
    import tensorflow as tf
    z = tf.signal.stft(waveform, frame_length=n_fft, frame_step=hop, pad_end=True)
    return tf.abs(z)

def logmel_tf(waveform):
    import tensorflow as tf
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
    import torchlibrosa as tl
    if 'tl_stft' not in cache:
        cache['tl_stft'] = tl.Spectrogram(n_fft=n_fft, hop_length=hop, power=1).cuda()
    
    return cache['tl_stft'].to(waveform.device)(waveform)[:, 0, :, :]

def logmel_tl(waveform):
    import torchlibrosa as tl
    if 'tl_mel' not in cache:
        cache['tl_mel'] = torch.nn.Sequential(
            tl.Spectrogram(n_fft=n_fft, hop_length=hop, power=1),
            tl.LogmelFilterBank(n_fft=n_fft, sr=sample_rate, n_mels=n_mels, fmin=0.0, fmax=0.5*sample_rate, is_log=False)
        ).cuda()

    melspectrogram = cache['tl_mel'].to(waveform.device)(waveform)
    return torch.log1p(melspectrogram)[:, 0, :, :]

def stft(waveform):
    if 'nn_stft' not in cache:
        cache['nn_stft'] = features.STFT(n_fft=2*n_mels-1, hop_length=hop, fmin=0, fmax=0.5*sample_rate, sr=sample_rate, output_format='Magnitude').cuda()
    
    return cache['nn_stft'].to(waveform.device)(waveform).permute(0, 2, 1)

def logstft(waveform):
    return torch.log1p(stft(waveform))

def logmel(waveform):
    if 'nn_mel' not in cache:
        cache['nn_mel'] = features.MelSpectrogram(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, power=1, fmin=0, fmax=0.5*sample_rate, hop_length=hop, verbose=False).cuda()

    melspectrogram = cache['nn_mel'].to(waveform.device)(waveform).permute(0, 2, 1)
    return torch.log1p(melspectrogram)

def find_bin_size(n_bins, fmin, sr):
    bin_sizes = np.arange(n_bins//20, n_bins, dtype=np.int32) # max: 20 octaves
    fmaxs = []
    octs = []
    for bins_per_octave in bin_sizes:
        n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        octs.append(n_octaves)
        
        # Calculate the lowest frequency bin for the top octave kernel
        fmin_t = fmin * 2 ** (n_octaves - 1)
        remainder = n_bins % bins_per_octave

        # Calculate the top bin frequency
        if remainder == 0:
            fmax_t = fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            fmax_t = fmin_t * 2 ** ((remainder - 1) / bins_per_octave)

        fmaxs.append(fmax_t)

    assert any(np.array(fmaxs) <= sr / 2), 'Could not find bin size'

    ibest = fmaxs.index(max([f for f in fmaxs if f <= sr/2]))
    return bin_sizes[ibest], octs[ibest]

def cqt1k(waveform):
    return cqt(waveform, yres=1024)

def cqt2k(waveform):
    return cqt(waveform, yres=2048)

def cqt(waveform, yres=512):
    key = f'nn_cqt_{yres}'
    if key not in cache:
        fmin = 0.5*32.7
        bin_size, n_oct = find_bin_size(n_bins=yres, fmin=fmin, sr=sample_rate)
        hop = max(512, 2**(n_oct-1))
        cache[key] = features.CQT2010v2(
            sr=sample_rate, n_bins=yres, bins_per_octave=bin_size, fmin=fmin, fmax=0.5*sample_rate, hop_length=hop).cuda()

    cqt = cache[key].to(waveform.device)(waveform).permute(0, 2, 1)
    return cqt

def comp_fwd(waveform):
    import tensorflow as tf
    wftf = tf.constant(waveform)

    # Spectrogram comparison
    mag1 = stft_tf(wftf)
    mag2 = stft_pt_tl(waveform)
    mag3 = stft(waveform)

    fig, ax = plot_comp([mag1, mag2, mag3], ['tf.signal.stft', 'tl.Spectrogram', 'nnAudio.STFT'], y_scale='log')
    fig.set_size_inches(30/2.54, 18/2.54)
    plt.savefig('stft_comp.png')
    plt.close('all')
    
    # Mel-scaled spectrogram comparison
    ms1 = logmel_tf(wftf)
    ms2 = logmel_tl(waveform)
    ms3 = logmel(waveform)

    s1_scaled = ms1 / tf.reduce_max(ms1)
    s2_scaled = ms2 / ms2.max()
    s3_scaled = ms3 / ms3.max()
    
    fig, ax = plot_comp([s1_scaled, s2_scaled, s3_scaled], ['tensorflow', 'torchlibrosa', 'nnAudio'], y_scale='mel')
    fig.set_size_inches(30/2.54, 18/2.54)
    plt.savefig('mel_comp.png')
    plt.close('all')

def comp_bwd(waveform, title):
    audio_io.write(_to_numpy(torch.cat(waveform.unbind(0))), f'{title}_orig.mp3', bitrate='256k')
    
    # Griffin-Lim
    spec = stft(waveform).permute(0, 2, 1).detach().cuda()
    w1 = features.Griffin_Lim(2*spec.shape[1]-1, hop_length=hop, device='cuda', n_iter=128)(spec)
    w1 = torch.cat(w1.unbind(0)).cpu().numpy()
    audio_io.write(w1, f'{title}_stft_griffinlim_{spec.shape[1]}x{spec.shape[2]}.mp3', bitrate='256k')
    #play_audio(w1, blocking=False)
    
    # Optimization-based
    for func in [cqt2k, logmel, stft, logstft]:
        device = 'cuda'
        steps = 300
        B = 1 if 'cqt' in func.__name__ else 2 # LBFGS scales poorly with batch size
        
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

                print('{} {}/{}: Img loss = {:.2e}, waveform loss: {:.2e}'.format(func.__name__, iteration, steps, loss.item(), waveform_loss.item()))
                return loss
            
            if isinstance(opt, optim.LBFGS):
                opt.step(closure)
            else:
                for _ in range(steps):
                    opt.step(closure)
            
            for c in noise:
                res.append(c.detach())

        w = torch.cat(res).detach().cpu().numpy()
        H, W = target.shape[1:]
        audio_io.write(w, f'{title}_{func.__name__}_lbfgs_{H}x{W}.mp3', bitrate='256k')
        #play_audio(w, blocking=False)
    
    print('Done')

if __name__ == '__main__':
    fname, offset = ('data/wav/shakuhachi.wav', 0)
    #fname, offset = ('C:/Users/Erik/eye_of_the_storm.wav', 15)
    title = Path(fname).with_suffix('').name

    n_parts = 3
    waveform, sr = audio_io.read(fname, offset=offset, duration=n_parts*duration)
    assert sr == sample_rate

    # Round to integer number of parts
    n_parts = len(waveform) // num_samples
    waveform = waveform[:n_parts*num_samples]

    # Add batch dim
    waveform = np.stack([waveform[i*num_samples:(i+1)*num_samples] for i in range(n_parts)], axis=0)
    wfpt = torch.from_numpy(waveform).to('cpu')

    # Generate parts for running inference
    for i, wf in enumerate(logmel(wfpt)):
        np.save(f'{title}_{i}.npy', wf.cpu().numpy())

    # Forward mode
    #comp_fwd(wfpt)

    # Reconstruction
    comp_bwd(wfpt, title)

    print('Done')