
import torch
import torch.nn.functional as F
import torch.optim as optim

import librosa as lr
import librosa.display
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt


plt.ion()

def show_cqt(cqt):
    plt.clf()
    librosa.display.specshow(lr.amplitude_to_db(cqt, ref=np.max),
                        sr=sample_rate, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)

cqt_filter_fft = lr.constantq.__cqt_filter_fft

class PseudoCqt():
    """A class to compute pseudo-CQT with Pytorch.
    Written by Keunwoo Choi
    Modified by Erik Härkönen, 2019
    API (+implementations) follows librosa (https://librosa.github.io/librosa/generated/librosa.core.pseudo_cqt.html)
    
    Usage:
        src, _ = librosa.load(filename)
        src_tensor = torch.tensor(src)
        cqt_calculator = PseudoCqt()
        cqt_calculator(src_tensor)
        
    """
    def __init__(self, sr=22050, hop_length=512, fmin=None, n_bins=84,
               bins_per_octave=12, tuning=0.0, filter_scale=1,
               norm=1, sparsity=0.01, window='hann', scale=True,
               pad_mode='reflect'):
        
        assert scale
        assert window == "hann"
        if fmin is None:
            fmin = 2 * 32.703195 # note_to_hz('C2') because C1 is too low

        if tuning is None:
            tuning = 0.0  # let's make it simple
        
        fft_basis, n_fft, _ = cqt_filter_fft(sr, fmin, n_bins, bins_per_octave,
                                               tuning, filter_scale, norm, sparsity,
                                               hop_length=hop_length, window=window)
        
        # Convert sparse to dense. (n_bins, n_fft)
        self.npdtype = np.float32
        fft_basis = np.abs(fft_basis.astype(dtype=self.npdtype)).todense()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.scale = scale
        self.fft_basis = torch.tensor(fft_basis).to(device)  # (n_freq, n_bins)
        self.window = torch.hann_window(self.n_fft).to(device)
        self.sqrt_epsilon = 1e-12
    
    def __call__(self, y):
        return self.forward(y)
    
    def forward(self, y):
        D_torch = torch.stft(y, self.n_fft, hop_length=self.hop_length, window=self.window).pow(2).sum(-1)  # n_freq, time
        D_torch = torch.sqrt(D_torch + self.sqrt_epsilon)  # without EPS, backpropagating through CQT can yield NaN.
        # Project onto the pseudo-cqt basis
        C_torch = torch.matmul(self.fft_basis, D_torch)  # n_bins, time
        C_torch /= np.sqrt(self.n_fft)  # because `scale` is always True
        return C_torch


def sonify(spectrogram, num_samples, transform_op_fn):
    # Start optimization from Gaussian noise
    noise = torch.tensor(np.random.normal(scale=1e-1, size=[num_samples]), dtype=torch.float32, requires_grad=True)

    # Learning rate schedule for Adam
    lr_schedule = lambda a: (1 - a)**1.5 * 0.015
    plt.plot(lr_schedule(np.linspace(0, 1, 100)))
    plt.title('Learning rate schedule')
    plt.show()
    plt.pause(0.5)
    
    max_steps = 3000
    #optimizer = optim.LBFGS(params=[noise], lr = 0.1, max_iter=max_steps, tolerance_change=0, tolerance_grad=0)
    optimizer = optim.Adam([noise], lr = lr_schedule(0))

    iteration = 0
    def closure():
        nonlocal iteration
        iteration += 1

        # Update learning rate
        for g in optimizer.param_groups:
            g['lr'] = lr_schedule(iteration / max_steps)

        optimizer.zero_grad()
        noise_in = noise.to(device)
        x = transform_op_fn(noise_in)
        y = spectrogram
        loss = F.mse_loss(x, y)
        loss.backward()
        
        #if iteration % 100 == 0:
        #    show_cqt(x.cpu().detach().numpy())

        print('{}/{}: Loss = {:.2e}'.format(iteration, max_steps, loss.item()))
        return loss
    
    if isinstance(optimizer, optim.LBFGS):
        optimizer.step(closure)
    else:
        for i in range(max_steps):
            optimizer.step(closure)

    cqt = transform_op_fn(noise.to(device)).cpu().detach().numpy()
    waveform = noise.detach().numpy()
    show_cqt(cqt)

    return waveform


sample_rate = 44100
duration = 5.0
num_samples = int(sample_rate*duration)
device = torch.device('cuda:0')
waveform, sr = lr.load('test_out.wav', sr=sample_rate, duration=duration)
assert len(waveform) == num_samples and sample_rate == sr

cqt_calculator = PseudoCqt()
def cqt_magnitude(x):
    return torch.abs(cqt_calculator(x))

src_tensor = torch.tensor(waveform).to(device)
C_mag = cqt_magnitude(src_tensor)
print('CQT shape:', C_mag.shape)

plt.figure()
reconstr = sonify(C_mag, num_samples, cqt_magnitude)

# Spare your ears
maxval = max(abs(reconstr.min()), abs(reconstr.max()))
if maxval > 1.0:
    reconstr /= maxval

sd.play(reconstr, sample_rate, blocking=True)

print('Done')
