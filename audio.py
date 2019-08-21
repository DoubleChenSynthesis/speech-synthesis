#-*-coding:utf-8-*-

import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from hyperparameters import hyperparameters

def load_wav(path):
    return librosa.core.load(path,sr=hyperparameters.sample_rate)[0]

def save_wav(wav,path):
    wav = wav / np.abs(wav).max()*0.999
    f1 = 0.5 * 32767 / max(0.01,np.max(np.abs(wav)))#防溢出
    f2 = np.sign(wav) * np.power(np.abs(wav),0.8)
    wav = f1 * f2
    firwin = signal.firwin(hyperparameters.num_freq,[hyperparameters.fmin,hyperparameters.fmax],pass_zero=False,fs=hyperparameters.sample_rate)
    wav = signal.convolve(wav,firwin)
    wavfile.write(path,hyperparameters.sample_rate,wav.astype(np.int16))

def trim_silence(wav):
  return librosa.effects.trim(wav, top_db= 60, frame_length=512, hop_length=128)[0]


def preemphasis(x):
  return signal.lfilter([1, -hyperparameters.preemphasis], [1], x)


def inv_preemphasis(x):
  return signal.lfilter([1], [1, -hyperparameters.preemphasis], x)


def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - hyperparameters.ref_level_db
  return _normalize(S)


def inv_spectrogram(spectrogram):
    #光谱图转换为波形
  S = _db_to_amp(_denormalize(spectrogram) + hyperparameters.ref_level_db)  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** hyperparameters.power))          # Reconstruct phase


def inv_spectrogram_tensorflow(spectrogram):
    #它不会反转预加重。调用者应该在运行图形之后调用inv_prestress on输出。
  S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hyperparameters.ref_level_db)
  return _griffin_lim_tensorflow(tf.pow(S, hyperparameters.power))


def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hyperparameters.ref_level_db
  return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hyperparameters.sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)


def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(hyperparameters.griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y


def _griffin_lim_tensorflow(S):
  '''TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
  '''
  with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(hyperparameters.griffin_lim_iters):
      est = _stft_tensorflow(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)


def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
  _, hop_length, win_length = _stft_parameters()
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(stfts):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters():
  n_fft = (hyperparameters.num_freq - 1) * 2
  hop_length = int(hyperparameters.frame_shift_ms / 1000 * hyperparameters.sample_rate)
  win_length = int(hyperparameters.frame_length_ms / 1000 * hyperparameters.sample_rate)
  return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
  n_fft = (hyperparameters.num_freq - 1) * 2
  assert hyperparameters.fmax < hyperparameters.sample_rate // 2
  return librosa.filters.mel(hyperparameters.sample_rate, n_fft, n_mels=hyperparameters.num_mels, fmin=hyperparameters.fmin, fmax=hyperparameters.fmax)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize(S):
  # symmetric mels
  return 2 * hyperparameters.max_abs_value * ((S - hyperparameters.min_level_db) / -hyperparameters.min_level_db) - hyperparameters.max_abs_value

def _denormalize(S):
  # symmetric mels
  return ((S + hyperparameters.max_abs_value) * -hyperparameters.min_level_db) / (2 * hyperparameters.max_abs_value) + hyperparameters.min_level_db

def _denormalize_tensorflow(S):
  # symmetric mels
  return ((S + hyperparameters.max_abs_value) * -hyperparameters.min_level_db) / (2 * hyperparameters.max_abs_value) + hyperparameters.min_level_db
