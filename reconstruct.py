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
batch = 30
target_shape = (1, 512, 512)
batch_shape = (batch, target_shape[1], target_shape[2])

# Target spectrogram
target_spectrogram = tf.Variable(np.zeros(batch_shape), dtype=tf.float32, trainable=False)
target_assign_placeholder = tf.placeholder(tf.float32, shape=batch_shape)
target_assign_op = tf.assign(target_spectrogram, target_assign_placeholder)

# Initial noise
noise = tf.Variable(np.zeros([batch * num_samples]), dtype=tf.float32)
noise_assign_placeholder = tf.placeholder(tf.float32, shape=[batch * num_samples])
noise_assign_op = tf.assign(noise, noise_assign_placeholder)

# waveform to mel-scaled stft
def logmel(waveform):
    z = tf.contrib.signal.stft(waveform, frame_length=2*2048, frame_step=500)
    magnitudes = tf.abs(z)
    filterbank = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=512, #80
        num_spectrogram_bins=magnitudes.shape[-1].value,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=0.5*sample_rate) #8k
    melspectrogram = tf.tensordot(magnitudes, filterbank, 1)
    return tf.log1p(melspectrogram)[:, :target_shape[1], :target_shape[2]]

def evaluate(input_tensor):
    minibatch = tf.reshape(input_tensor, (batch, num_samples))
    x = logmel(minibatch)
    y = target_spectrogram

    x = tf.expm1(x)
    y = tf.expm1(y)

    loss = tf.losses.mean_squared_error(x, y)
    return (loss, *tf.gradients(loss, input_tensor))

use_scipy = False
if use_scipy:
    loss = evaluate(noise)[0]
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        loss=loss,
        var_list=[noise],
        tol=1e-16,
        method='L-BFGS-B',
        options={ 'maxiter': 500, 'disp': False }
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

# Pad to minibatch size
files = sys.argv[1:]
missing = batch - len(files) % batch
files = files + missing * [files[-1]]

for f in range(0, len(files), batch):
    print('Processing batch', f // batch, 'of', len(files) // batch)

    paths = files[f:f+batch]

    spectrograms = np.zeros(batch_shape, dtype=np.float32)
    for i, p in enumerate(paths):
        spectrogram = np.load(p)
        if len(spectrogram.shape) == 2:
            spectrogram = np.expand_dims(spectrogram, axis=0)
        if spectrogram.shape != target_shape:
            print('\nERROR: File {} is of wrong shape: got {}, expected {}.\n'.format(p, spectrogram.shape, target_shape))
            spectrogram = np.zeros(target_shape, dtype=np.float32)
        spectrograms[i] = spectrogram

    initial_noise = np.random.normal(scale=1e-6, size=[batch * num_samples]) # tfp needs 1d

    # Assign spectrogram and noise
    sess.run(target_assign_op, feed_dict={ target_assign_placeholder: spectrograms })
    sess.run(noise_assign_op, feed_dict={ noise_assign_placeholder: initial_noise })

    import time
    ts = time.time()
    
    if use_scipy:
        optimizer.minimize(sess)
        print('Final loss: {:.2e}'.format(loss.eval()))
        reconstructed_waveforms = sess.run(noise)
    else:
        res = sess.run(retval)
        print('Final loss: {:.2e}'.format(res.objective_value))
        reconstructed_waveforms = np.reshape(res.position, (batch, num_samples))

    delta = time.time() - ts
    print('Done in {:.2f}s ({:.2f} s/img)'.format(delta, delta / batch))

    for wf, path in zip(reconstructed_waveforms, paths):
        #sd.play(wf, sample_rate, blocking=True)
        out_path = Path(path).with_suffix('.wav')
        lr.output.write_wav(str(out_path), wf, sample_rate, norm=False)

print('Done')
