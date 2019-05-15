import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd  # conda version only supports python 3.6
import numpy as np
import subprocess
import shutil
from pathlib import Path
from sf2utils.sf2parse import Sf2File #pip install sf2utils

SYNTH_BIN = 'external/windows/timidity++/timidity.exe' #shutil.which('timidity')
SYNTH_CFG = 'data/soundfont/config.cfg' # points to currently selected soundfont

# Create a temporary config file pointing to the correct soundfont
def select_midi_soundfont(name, instrument=''):
    matches = sorted(Path('./data/soundfont/').glob('**/' + name))
    if len(matches) == 0:
        raise Exception('Could not find soundfont: ' + name)
    elif len(matches) > 1:
        print('Multiple matching soundfonts:', matches)
        print('Using first match')

    fontpath = matches[0]

    with open(fontpath.resolve(), 'rb') as sf2_file:
        sf2 = Sf2File(sf2_file)
        preset_num = sf2.presets[0].preset
        for preset in sf2.presets:
            if preset.name.lower() == instrument.lower():
                preset_num = preset.preset
            if preset.name != 'EOP':
                print('Preset {}: {}'.format(preset.preset, preset.name))
        print('Using preset', preset_num)

    with open(SYNTH_CFG, 'w') as f:
        config = "dir {}\nbank 0\n0 %font \"{}\" 0 {} amp=100".format(fontpath.parent.resolve(), name, preset_num)
        f.write(config)

def midi2wav(file):
    outpath = 'test_out.wav'
    
    cmds = [
        SYNTH_BIN, '-c', SYNTH_CFG, file,
        '-Od', '--reverb=d' '--noise-shaping=4',
        '-EwpvseToz', '-f', '-A100', '-Ow',
        '-o', outpath
    ]
    
    print('Converting midi to wav...', end='', flush=True)
    current_subprocess = subprocess.Popen(cmds) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    current_subprocess.wait()
    print('done')

def play_midi(file):
    cmds = [SYNTH_BIN, '-c', SYNTH_CFG, file]
    current_subprocess = subprocess.Popen(cmds) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    current_subprocess.wait()

if True:
    # Config-1k
    target_shape = (512, 1024)
    hop_length = 64 * 1       # y-reso (time)
    bins_per_octave = 12 * 6  # x-reso (freq)
    target_sr = 22050         # y-reso (time)
    time_window = 3.0         # y-reso
else:
    # Config-2k
    target_shape = (1024, 2048)
    hop_length = 64 * 1        # y-reso (time)
    bins_per_octave = 12 * 6   # x-reso (freq)
    target_sr = 22050          # y-reso (time)
    time_window = 3.0          # y-reso


# Instument remapping readme:
#http://timbrechbill.com/saxguru/faqSf2.html#q10

#select_midi_soundfont('068_Florestan_Woodwinds.sf2', 'Oboe')
#select_midi_soundfont('vibraphone-sustain-ff.sf2')
#select_midi_soundfont('Milton_Pan_flute.sf2')
select_midi_soundfont('Seinfeld__KorgM1__SlapBass.sf2', 'Little Swell Choir')

#select_midi_soundfont('SoundBlasterPiano.sf2')
#select_midi_soundfont('spanish-classical-guitar.sf2')
#select_midi_soundfont('grand-piano-YDP-20160804.sf2')
#select_midi_soundfont('EGuitarClean.sf2')
#select_midi_soundfont('SoftStrings.sf2')

midi2wav('data/midi/moonlight_3rd.mid')
audio, offset = ('test_out.wav', 0.0)

y, sr = librosa.load(audio, sr=target_sr, offset=offset, duration=time_window)
print(y.shape, sr)

# Scale if waveform outside of range [-1, 1]
absmax = max(abs(y.min()), abs(y.max()))
if absmax > 1.0:
    y /= absmax
    print('Scaled waveform by', absmax)

assert target_sr == sr, 'Imported audio has wrong sampling rate'
assert y.min() >= -1.0 and y.max() <= 1.0, 'Audio in incorrect range'

sd.play(y, sr, blocking=True)

#def reshape_spectrogram(S):
#	missing_x = max(0, target_shape[0] - S.shape[0])
#	missing_y = max(0, target_shape[1] - S.shape[1])
#	S = np.pad(S, (missing_x, missing_y), mode='constant')
#	S = S[0:target_shape[0], 0:target_shape[1]] # clip
#	return S
#
## CQT
#C = librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=7*bins_per_octave, bins_per_octave=bins_per_octave)
#C_mag = np.abs(C) # CQT magnitude
#print('CQT shape:', C_mag.shape)
##C = reshape_spectrogram(C)
#print('CQT new shape:', C_mag.shape)
#
## Generate white noise phase information
##excitation = 0.1 * np.random.randn(y.shape[0])
##C_noise = librosa.cqt(y=excitation, sr=sr, hop_length=hop_length, n_bins=7*bins_per_octave, bins_per_octave=bins_per_octave)
##phase = np.imag(C_noise) # cartesian, not polar
##
### Add phase, preserve manitude
##phase_ang = np.arcsin(phase / C_mag)
##C_complex = C_mag*np.cos(phase_ang) + 1j * phase
##C_new_mag = np.sqrt(np.real(C_complex)**2 + np.imag(C_complex)**2) # should match C_mag
#
#
## TODO: Negative values from CNN?
#
#y_hat = librosa.icqt(C=C_mag.astype(np.complex), sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave)
#
#plt.figure(figsize=(14, 2*5))
#plt.suptitle('{}kHz'.format(sr//1000))
#
#plt.subplot(2,1,1)
#librosa.display.waveplot(y, sr=sr)
#plt.subplot(2,1,2)
#librosa.display.waveplot(y_hat, sr=sr)
#
#plt.show()
#
## Play input
#sd.play(y, sr, blocking=True)
#sd.play(y_hat, sr, blocking=True)
#sd.play(y_hat_cheat, sr, blocking=True)

#from scipy.io.wavfile import write
#write('input.wav', sr, y.astype(np.float32))
#write('recon_zero_phase.wav', sr, y_hat.astype(np.float32))
#write('recon_cheat.wav', sr, y_hat_cheat.astype(np.float32))
#write('recon_cast.wav', sr, y_hat_cast.astype(np.float32))
