
# generate spectrogram dataset from midi
# midi files from dataset maestro-v1.0.0 from https://magenta.tensorflow.org/datasets/maestro (automatically downloaded if not present)

import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd  # conda version only supports python 3.6
from sf2utils.sf2parse import Sf2File # pip install sf2utils
import numpy as np
import subprocess
from pathlib import Path
import urllib.request
import os
import zipfile
import shutil
import random
import threading
import time

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
def select_midi_soundfont(name, instrument='default'):
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
            #if preset.name != 'EOP':
            #    print('Preset {}: {}'.format(preset.preset, preset.name))
        print('Using preset', preset_num)
    
    cfgpath = fontpath.with_suffix('.'+instrument+'.cfg')
    with open(cfgpath, 'w') as f:
        config = "dir {}\nbank 0\n0 %font \"{}\" 0 {} amp=100".format(fontpath.parent.resolve(), name, preset_num)
        f.write(config)
    return cfgpath

def midi2wav(file, outpath, cfg):
    cmds = [
        SYNTH_BIN, '-c', str(cfg), str(file),
        '-Od', '--reverb=g,25' '--noise-shaping=4'
        '-EwpvseToz', '-f', '-A100', '-Ow',
        '-o', str(outpath)
    ]
    
    #print('Converting midi to wav...', end='', flush=True)
    return subprocess.call(cmds, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

sample_rate = 44100
duration = 6.0
num_samples = int(sample_rate*duration)

# waveform to mel-scaled stft
def logmel(waveform):
    z = tf.contrib.signal.stft(waveform, frame_length=4096, frame_step=500)
    magnitudes = tf.abs(z)
    filterbank = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=512, #80
        num_spectrogram_bins=magnitudes.shape[-1].value,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=0.5*sample_rate) #8k
    melspectrogram = tf.tensordot(magnitudes, filterbank, 1)
    return tf.log1p(melspectrogram)[:512, :512]

# suppress tf output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)
if type(tf.contrib) != type(tf): tf.contrib._warning = None

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # doesn't make sense to run the small-ish stft on GPU

# dataset parameters
spectrogram_count = 20000
block_size = 20 # how many extracts taken from each track

# render matching audio for each of these soundfonts
instruments = {
    #'piano': ('grand-piano-YDP-20160804.sf2', ''),
    'piano': ('SoundBlasterPiano.sf2', ''),
    #'flute': ('Milton_Pan_flute.sf2', ''),
    'guitar': ('spanish-classical-guitar.sf2', ''),
    #'harp' : ('Roland_SC-88.sf2', 'Harp'),
    #'kalimba' : ('Roland_SC-88.sf2', 'Kalimba'),
    #'pan' : ('Roland_SC-88.sf2', 'Pan flute')
}
with tf.Session() as sess:

    # construct the tf graph
    waveform_data = tf.placeholder(dtype='float32', shape=(num_samples))
    output = logmel(waveform_data)

    # initialize instruments configurations and result files
    cfgs = []
    for instrument in instruments:
        os.makedirs('data/spectrogram', exist_ok=True)
        dataFile = Path('data/spectrogram/maestro_'+instrument+'.npy')
        if dataFile.exists():
            print(str(dataFile), 'already exists!')
            #continue
        else:
            np.save(str(dataFile), np.zeros((spectrogram_count, 512,512), dtype='float32'))
        cfgs.append(select_midi_soundfont(*instruments[instrument]))
    
    if len(cfgs) == len(instruments):
        # render the given midi file with the given instruments, generate spectrograms and save
        def renderMidi(i, f, cfgs):
            print(str(f), flush=True)
            min_len = None
            # render the waveform files
            for (j, instrument) in enumerate(instruments):
                print(instrument, '...')
                # synthesize midi with timidity++, obtain waveform
                file = Path(f).with_suffix('.'+instrument+'.wav')
                if midi2wav(f, file, cfgs[j])!=0:
                    print('midi2wav failed!')
                    result[:,:,i*block_size:(i+1)*block_size] = np.zeros((result.shape[0], result.shape[1], block_size))
                    return
                cur_len = len(librosa.load(str(file), sr = sample_rate)[0])
                if j==0 or cur_len<min_len:
                    min_len = cur_len

            # turn waveforms into sets of spectrograms
            for instrument in instruments:
                # open waveform and result file
                waveform, _ = librosa.load(str(Path(f).with_suffix('.'+instrument+'.wav')), sr = sample_rate)
                fp = np.load(Path('data/spectrogram/maestro_'+instrument+'.npy'), mmap_mode='r+')
                # process extracts of the waveform at uniform intervals
                for (j, start) in enumerate(np.linspace(0, min_len-num_samples-1, num=block_size)):
                    layer = i*block_size+j
                    if layer>=spectrogram_count:
                        break
                    start = min(int(start), min_len-num_samples-1)
                    # output the spectrogram
                    fp[layer,:,:] = sess.run(output, feed_dict={waveform_data:waveform[start:start+num_samples]})
                    # print progress
                    print('\r', instrument, ' ', layer,'/',spectrogram_count,'done', flush=True,end="")
                print('')
                #write to disk
                del fp
            # cleanup; remove the waveforms
            for instrument in instruments:
                file = Path(f).with_suffix('.'+instrument+'.wav')
                os.remove(file)

        # gather the files and process
        files = []
        for f in find_files(setDir):
            if 'midi' in Path(f).suffix:
                files.append(f)
        for (i, f) in enumerate(files):
            if i*block_size<spectrogram_count:
                renderMidi(i, f, cfgs)
        print('done!')
