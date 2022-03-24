import torch
import torch.optim
import torch.nn.functional as F
import numpy as np
import sys
from constants import *
from port_spectrogram import logmel
import audio_io

# Input spectrogram npys
files = sys.argv[1:]

device = 'cuda'
steps = 300
B = 2 # LBFGS scales poorly with batch size
func = logmel

specs = np.stack([np.load(p) for p in files], axis=0)
specs = torch.tensor(specs, dtype=torch.float32)

res = []
for i in range(0, len(files), B):
    target = specs[i:i+B].to(device)
    noise = torch.tensor(np.random.normal(scale=1e-1, size=[target.shape[0], num_samples]), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.LBFGS(params=[noise], lr=0.75, max_iter=steps, tolerance_change=0, tolerance_grad=0)

    iteration = 0
    def closure():
        global iteration
        iteration += 1

        # TOOD???
        #x = tf.expm1(x)
        #y = tf.expm1(y)

        opt.zero_grad()
        loss = F.mse_loss(func(noise), target)
        loss.backward()

        print('{} {}/{}: loss = {:.2e}'.format(func.__name__, iteration, steps, loss.item()))
        return loss
    
    opt.step(closure)
    
    for j, waveform in enumerate(noise.detach().cpu().numpy()):
        audio_io.write(waveform, f'{i+j}.mp3', bitrate='256k')
        #play_audio(waveform, blocking=False)

print('Done')
