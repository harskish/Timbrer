sample_rate = 44100
duration = 6.0
num_samples = int(sample_rate*duration)
n_mels = 512 # target y resolution (frequency)
hop = 517 # produces time resolution 512 (no audible seams)
n_fft = 2*2048 # produces n_fft/2+1 intermediate y-res before mel scaling