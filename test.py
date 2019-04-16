import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd  # Windows: must install with pip (not conda)
import numpy as np
import subprocess
import shutil

SYNTH_BIN = 'external/windows/timidity++/timidity.exe' #shutil.which('timidity')	

def midi2wav(file):
	outpath = 'test_out.wav'
	
	cmds = [
		SYNTH_BIN, '-c', 'data/soundfont/8MBGMSFX.cfg', file,
		'-Od', '--reverb=d' '--noise-shaping=4',
		'-EwpvseToz', '-f', '-A100', '-Ow',
		'-o', outpath
	]
	
	print('Converting midi to wav...', end='', flush=True)
	current_subprocess = subprocess.Popen(cmds, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	current_subprocess.wait()
	print('done')

def play_midi(file):
	cmds = [
		SYNTH_BIN,
		'-c',
		'data/soundfont/8MBGMSFX.cfg',
		file,
	]
		
	current_subprocess = subprocess.Popen(cmds) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	current_subprocess.wait()

if True:
    # Config-1k
    hop_length = 64 * 1       # y-reso (time)
    bins_per_octave = 12 * 6  # x-reso (freq)
    target_sr = 22050         # y-reso (time)
    time_window = 3.0         # y-reso
else:
    # Config-2k
    hop_length = 64 * 1        # y-reso (time)
    bins_per_octave = 12 * 6   # x-reso (freq)
    target_sr = 22050          # y-reso (time)
    time_window = 3.0          # y-reso

midi2wav('data/midi/chopin_op25_12.mid')
audio, offset = ('test_out.wav', 0.0)
#audio, offset = ('data/shakuhachi.wav', 5.0)

y, sr = librosa.load(audio, sr=target_sr, offset=offset, duration=time_window)
print(y.shape, sr)

assert target_sr == sr, 'Imported audio has wrong sampling rate'
assert y.min() >= -1.0 and y.max() <= 1.0, 'Audio in incorrect range'

# CQT
C = librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=7*bins_per_octave, bins_per_octave=bins_per_octave)
y_hat = librosa.icqt(C=C, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave)

print('CQT size:', C.shape)
plt.figure(figsize=(14, 2*5))
plt.suptitle('{}kHz'.format(sr//1000))

plt.subplot(2,1,1)
librosa.display.waveplot(y, sr=sr)
plt.subplot(2,1,2)
librosa.display.waveplot(y_hat, sr=sr)

plt.show()

# Play input
sd.play(y, sr, blocking=True)
sd.play(y_hat, sr, blocking=True)

