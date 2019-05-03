
# generate spectrogram dataset from midi
# midi files from dataset maestro-v1.0.0 from https://magenta.tensorflow.org/datasets/maestro (automatically downloaded if not present)

import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd  # Windows: must install with pip (not conda)
import numpy as np
import subprocess
from pathlib import Path
import urllib.request
import os
import zipfile
import shutil
import random
import threading

from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf

setDir = 'data/midi/maestro'
SYNTH_BIN = 'external/windows/timidity++/timidity.exe'

# helper: find all possible files in a certain directory (and subdirectories)
def find_files(root):
    for d, dirs, files in os.walk(root):
        for f in files:
            yield os.path.join(d, f)

# print progress when downloading the dataset
def retrieveCallback(blocks_done, block_size, total_size):
    print('\r{:.0f}MB/{:.0f}MB'.format(blocks_done*block_size/(1024*1024), total_size/(1024*1024)), ' '*10, end = "", flush=True)
                
# if set not found: download and extract
if not os.path.isdir(setDir):
    url = 'https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip'
    urllib.request.urlretrieve(url, 'data/midi/maestro.zip', retrieveCallback)
    print('\nExtracting dataset..')
    with zipfile.ZipFile('data/midi/maestro.zip', 'r') as zip:
        zip.extractall(setDir+'/tmp')
    
    # simplify: remove extra files and flatten folder structure
    print('Flattening/cleaning dataset..')
    for f in find_files(setDir+'/tmp'):
        if('midi' in Path(f).suffix):
            shutil.move(f, setDir + "/" + Path(f).name)
        else:
            os.remove(f)
    shutil.rmtree(setDir+'/tmp')


# Create a temporary config file pointing to the correct soundfont
def select_midi_soundfont(name):
    matches = sorted(Path('./data/soundfont/').glob('**/' + name))
    if len(matches) == 0:
        raise Exception('Could not find soundfont: ' + name)
    elif len(matches) > 1:
        print('Multiple matching soundfonts:', matches)
        print('Using first match')
    fontpath = matches[0]
    cfgpath = fontpath.with_suffix('.cfg')
    with open(cfgpath, 'w') as f:
        config = "dir {}\nsoundfont \"{}\" amp=100%".format(fontpath.parent.resolve(), name)
        f.write(config)
    return cfgpath

def midi2wav(file, outpath, cfg):
    cmds = [
        SYNTH_BIN, '-c', str(cfg), str(file),
        '-Od', '--reverb=d' '--noise-shaping=4',
        '-EwpvseToz', '-f', '-A100', '-Ow',
        '-o', str(outpath)
    ]
    
    #print('Converting midi to wav...', end='', flush=True)
    current_subprocess = subprocess.Popen(cmds, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    current_subprocess.wait()
    #print('done')

sample_rate = 44100
duration = 6.0
num_samples = int(sample_rate*duration)

# waveform to mel-scaled stft
def logmel(waveform):
    z = tf.contrib.signal.stft(waveform, frame_length=2048, frame_step=1024)
    magnitudes = tf.abs(z)
    filterbank = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=256, #80
        num_spectrogram_bins=magnitudes.shape[-1].value,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=0.5*sample_rate) #8k
    melspectrogram = tf.tensordot(magnitudes, filterbank, 1)
    return tf.log1p(melspectrogram)[:256, :256]

# suppress tf output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)
if type(tf.contrib) != type(tf): tf.contrib._warning = None

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # doesn't make sense to run the small-ish stft on GPU

# dataset parameters
spectrogram_count = 200
block_size = 10 # how many extracts taken from each track

# render matching audio for each of these soundfonts
instruments = {
    'piano': 'grand-piano-YDP-20160804.sf2',
    'guitar': 'spanish-classical-guitar.sf2'
}
with tf.Session():
    for instrument in instruments:
        
        dataFile = Path('data/spectrogram/maestro_'+instrument+'.npy')
        if dataFile.exists():
            print(str(dataFile), 'already exists -- skipping')
            continue
        
        cfg = select_midi_soundfont(instruments[instrument])
        
        # write results to this tensor
        result = np.zeros((256, 256, spectrogram_count), dtype=np.float32)
        
        done = 0

        print('beginning', instrument)

        def renderBlock(i, f):
            global lock, done, sessions
            # synthesize midi with timidity++, obtain waveform
            file = Path(f).with_suffix('.'+instrument+'.wav')
            print(str(f), flush=True)
            midi2wav(f, file, cfg)
            waveform, _ = librosa.load(str(file), sr = sample_rate)
            #assert sr == sample_rate
            os.remove(file)

            # generate spectrograms from extracts of the track at uniform intervals
            for (j, start) in enumerate(np.linspace(0, len(waveform)-num_samples-1, num=block_size)):
                layer = i*block_size+j
                if layer>=spectrogram_count:
                    break
                start = min(int(start), len(waveform)-num_samples-1)
                result[:,:,layer] = logmel(waveform[start:start+num_samples]).eval()
                # print progress
                done += 1
                print('\rspectrogram', done,'/',spectrogram_count,'done', flush=True,end="")
            print('')
        files = []
        for f in find_files(setDir):
            if 'midi' in Path(f).suffix:
                files.append(f)
        for (i, f) in enumerate(files):
            if i*block_size<spectrogram_count:
                renderBlock(i, f)
        
        print('saving..')
        np.save(str(dataFile), result)
