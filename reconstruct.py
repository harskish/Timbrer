import torch
import torch.optim
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from constants import num_samples
from port_spectrogram import logmel
import audio_io
from tqdm import trange
import argparse

func = logmel

parser = argparse.ArgumentParser(description='Waveform reconstruction from spectrograms')
parser.add_argument('paths', help='Path(s) to (directories containing) spectrograms', type=Path, nargs='+')
parser.add_argument('--steps', help='L-BFGS optimization steps (default: %(default)d)', type=int, default=70)
parser.add_argument('--batch', help='L-BFGS batch size (default: %(default)d)', type=int, default=2) # LBFGS scales poorly with batch size
parser.add_argument('--device', help='Compute device (default: %(default)s)', type=str, default='cuda')
parser.add_argument('--bitrate', help='Output mp3 bitrate in Kbps (default: %(default)dk)', type=int, default=256)
parser.add_argument('--verbose', help='Verbose mode (print reconstruction loss)', action='store_true')
args = parser.parse_args()

files = []
for p in args.paths:
    if p.is_dir():
        files = files + list(p.glob('*.npy'))
    elif p.suffix == '.npy':
        files.append(p)
    
assert files, 'No spectrograms found'

specs = np.stack([np.load(p).squeeze() for p in files], axis=0)
specs = torch.tensor(specs, dtype=torch.float32)

res = []
for i in trange(0, len(files), args.batch):
    target = specs[i:i+args.batch].to(args.device)
    noise = torch.tensor(np.random.normal(scale=1e-1, size=[target.shape[0], num_samples]), dtype=torch.float32, device=args.device, requires_grad=True)
    opt = torch.optim.LBFGS(params=[noise], lr=0.7, max_iter=args.steps, tolerance_change=0, tolerance_grad=0)

    iteration = 0
    def closure():
        global iteration
        iteration += 1

        #x = torch.expm1(x)
        #y = torch.expm1(y)

        opt.zero_grad()
        loss = F.mse_loss(func(noise), target)
        loss.backward()

        if args.verbose:
            print('{} {}/{}: loss = {:.2e}'.format(func.__name__, iteration, args.steps, loss.item()))
        return loss
    
    opt.step(closure)
    
    for j, waveform in enumerate(noise.detach().cpu().numpy()):
        outpath = Path(files[i+j]).with_suffix('.mp3')
        audio_io.write(waveform, str(outpath), bitrate=f'{args.bitrate}k')

print('Done')
