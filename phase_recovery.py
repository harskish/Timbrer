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

def sonify(spectrogram, samples, transform_op_fn, logscaled=True):
    use_adam = True
    
    graph = tf.Graph()
    with graph.as_default():

        noise = tf.Variable(tf.random_normal([samples], stddev=1e-6))

        x = transform_op_fn(noise)
        y = spectrogram

        if logscaled:
            x = tf.expm1(x)
            y = tf.expm1(y)

        # Normalization results in lost amplitude (but better convergence?)
        #x = tf.nn.l2_normalize(x)
        #y = tf.nn.l2_normalize(y)

        loss = tf.losses.mean_squared_error(x, y)

        if use_adam:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            minimize_op = optimizer.minimize(loss, var_list=[noise])
        else:
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss=loss,
                var_list=[noise],
                tol=1e-16,
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'disp': True
                })

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        
        if use_adam:
            for i in range(5000):
                _, loss_val = session.run([minimize_op, loss])
                print('Iteration {}: loss = {:.3e}'.format(i + 1, loss_val))
        else:
            optimizer.minimize(session)
        
        print('Final loss:', loss.eval())
        waveform = session.run(noise)

    return waveform


sample_rate = 44100
duration = 5.0
num_samples = int(sample_rate*duration)
waveform, sr = lr.load('test_out.wav', sr=sample_rate, duration=duration)
assert len(waveform) == num_samples and sample_rate == sr

def logmel(waveform):
    z = tf.contrib.signal.stft(waveform, frame_length=8192*2, frame_step=1024)
    magnitudes = tf.abs(z)
    filterbank = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=1024, #80
        num_spectrogram_bins=magnitudes.shape[-1].value,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=0.5*sample_rate) #8k
    melspectrogram = tf.tensordot(magnitudes, filterbank, 1)
    return tf.log1p(melspectrogram)


with tf.Session():
    spectrogram_input = logmel(waveform).eval()
    np.save('test_mel_spectrogram.npy', spectrogram_input)

spectrogram = np.load('test_mel_spectrogram.npy') # restore from disk

plt.imshow(spectrogram)
plt.show()

print('Mel spectrogram shape:', spectrogram.shape)
reconstructed_waveform = sonify(spectrogram, num_samples, logmel)

sd.play(reconstructed_waveform, sample_rate, blocking=True)

print('Done')
