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
import tensorflow_probability as tfp
import librosa as lr
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Parameters of data set
sample_rate = 44100
duration = 6.0
num_samples = int(sample_rate*duration)
target_shape = (512, 512)

# Target spectrogram
target_spectrogram = tf.Variable(np.zeros(target_shape), dtype=tf.float32, trainable=False)
target_assign_placeholder = tf.placeholder(tf.float32, shape=target_shape)
target_assign_op = tf.assign(target_spectrogram, target_assign_placeholder)

# Initial noise
noise = tf.Variable(np.zeros([num_samples]), dtype=tf.float32)
noise_assign_placeholder = tf.placeholder(tf.float32, shape=[num_samples])
noise_assign_op = tf.assign(noise, noise_assign_placeholder)

use_scipy = True

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
    return tf.log1p(melspectrogram)[:target_shape[0], :target_shape[1]]

def evaluate(input_tensor):
    x = logmel(input_tensor)
    y = target_spectrogram

    x = tf.expm1(x)
    y = tf.expm1(y)

    loss = tf.losses.mean_squared_error(x, y)
    return loss, tf.gradients(loss, input_tensor)[0]

if use_scipy:
    loss = evaluate(noise)[0]
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        loss=loss,
        var_list=[noise],
        tol=1e-16,
        method='L-BFGS-B',
        options={ 'maxiter': 1000, 'disp': False }
    )
else:
    retval = tfp.optimizer.lbfgs_minimize(
        evaluate,
        noise,
        num_correction_pairs=10,
        tolerance=1e-08,
        max_iterations=500
    )

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for p in sys.argv[1:]:
    print('Processing', p)
    in_path = Path(p)

    spectrogram = np.load(str(in_path))[0]
    initial_noise = np.random.normal(scale=1e-6, size=[num_samples])

    # Assign spectrogram and noise
    sess.run(target_assign_op, feed_dict={ target_assign_placeholder: spectrogram })
    sess.run(noise_assign_op, feed_dict={ noise_assign_placeholder: initial_noise })

    import time
    ts = time.time()
    
    if use_scipy:
        optimizer.minimize(sess)
        print('Final loss: {:.2e}'.format(loss.eval()))
        reconstructed_waveform = sess.run(noise)
    else:
        res = sess.run(retval)
        print('Final loss: {:.2e}'.format(res.objective_value))
        reconstructed_waveform = res.position

    print('Done in', time.time() - ts, 'seconds')

    #sd.play(reconstructed_waveform, sample_rate, blocking=True)
    lr.output.write_wav(str(in_path.with_suffix('.wav')), reconstructed_waveform, sample_rate, norm=False)

print('Done')
