# # Signal reconstruction from spectrograms
# Reconstruct waveform from input spectrogram by iteratively minimizing a cost function between the spectrogram and white noise transformed into the exact same time-frequency domain.
# 
# Assuming 50% magnitude overlap and linearly spaced frequencies this reconstruction method is pretty much lossless in terms of audio quality, which is nice in those cases where phase information cannot be recovered.
# 
# Given a filtered spectrogram such as with a Mel filterbank, the resulting audio is noticeably degraded (particularly due to lost treble) but still decent.
# 
# The biggest downside with this method is that the iterative procedure is very slow (running on a GPU is a good idea for any audio tracks longer than 20 seconds) compared to just having an inverse transform at hand.
# 
# ## Reference
# - Decorsière, Rémi, et al. "Inversion of auditory spectrograms, traditional spectrograms, and other envelope representations." IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP) 23.1 (2015): 46-56.

# Original code by Carl Thomé at Github Gist:
# https://gist.github.com/carlthome/a4a8bf0f587da738c459d0d5a55695cd
# Modified by Erik Härkönen

import tensorflow as tf
import librosa as lr
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def sonify(spectrogram, samples, transform_op_fn, logscaled=True):
    graph = tf.Graph()
    with graph.as_default():

        noise = tf.Variable(tf.random_normal([samples], stddev=1e-6))

        x = transform_op_fn(noise)
        y = spectrogram

        if logscaled:
            x = tf.expm1(x)
            y = tf.expm1(y)

        loss = tf.losses.mean_squared_error(x, y)

        global_step = tf.Variable(0, trainable=False)
        start_learning_rate = 0.1
        opt_steps = 5000
        num_decays = 70
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps=opt_steps//num_decays, decay_rate=0.96, staircase=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        minimize_op = optimizer.minimize(loss, var_list=[noise], global_step=global_step)

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        
        for i in range(opt_steps):
            _, loss_val = session.run([minimize_op, loss])
            print('Iteration {}: loss = {:.3e}'.format(i + 1, loss_val))
        
        print('Final loss:', loss.eval())
        waveform = session.run(noise)

    return waveform

# Parameters of data set
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

for p in sys.argv[1:]:
    print('Processing', p)
    in_path = Path(p)

    spectrogram = np.load(str(in_path)) # restore from disk
    reconstructed_waveform = sonify(spectrogram[0], num_samples, logmel)

    #sd.play(reconstructed_waveform, sample_rate, blocking=True)
    lr.output.write_wav(str(in_path.with_suffix('.wav')), reconstructed_waveform, sample_rate, norm=False)

print('Done')
