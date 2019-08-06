#-*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.seq2seq import CustomHelper
import embedding
from embedding import prenet
import matplotlib
matplotlib.use('pdf')
import copy
import librosa
from hyperparameters import hyperparameters as hp
import numpy as np
from scipy import signal
import os,sys,io
import matplotlib.pyplot as plt
class decoder_prenet_wrapper(RNNCell):
    #prenet of RNN
    def __init__(self,cell,training):
        super(decoder_prenet_wrapper,self).__init__()
        self.cell = cell
        self.training = training
        @property
        def state_size(self):
            return self.cell.state_size

        @property
        def output_size(self):
            return self.cell.output_size

        def call(self,inputs,state):
            prenet_out = prenet(inputs,256,128,self.training,scope='decoder_prenet')
            return self.cell(prenet_out,state)

        def zero_state(self, batch_size, dtype):
            return self.cell.zero_state(batch_size,dtype)

class inference_helper(CustomHelper):
    #评估合成
    def __init__(self,batch_size,out_size):
        super(inference_helper,self).__init__(self.initialize_fn,self.sample_fn,self.next_inputs_fn)
        self.batch_size = batch_size
        self.out_size = out_size

    #重构输出
    def initialize_fn(self):
        return (tf.tile([False],[self.batch_size]),tf.zeros([self.batch_size,self.out_size],dtype=tf.float32))

    def sample_fn(self,time,outputs,state):
        return tf.zeros(self.batch_size,dtype=tf.int32)

    def next_inputs_fn(self, time, outputs, state, sample_ids, name=None):
        del time,sample_ids
        return (tf.tile([False],[self.batch_size]),outputs,state)



def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sample_rate)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.num_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sample_rate, hp.num_fft, hp.num_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def spectrogrom2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.grilimn_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.num_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def plot_alignment(alignment, gs, mode, path = None):
    """Plots the alignment
    alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    mode: "save_fig" or "with_return". "save_fig":save fig locally, "with_return":return plot for tensorboard
    """

    plt.imshow(alignment, cmap='hot', interpolation='nearest')
    plt.xlabel('Decoder Steps')
    plt.ylabel('Encoder Steps')
    plt.title('{} Steps'.format(gs))

    if mode == "save_fig":
        if path is not None:
            plt.savefig('{}/alignment_{}k.png'.format(path, gs // hp.save_model_step), format='jpg')
        else:
            print ("Warning! You need specify the saved path! The temporal path is {}".format(hp.logdir))
            plt.savefig('{}/alignment_{}k.png'.format(hp.logdir, gs // hp.save_model_step), format='jpg')

    elif mode == "with_return":
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot = tf.image.decode_png(buf.getvalue(), channels=4)
        plot = tf.expand_dims(plot,0)
        return plot

    else:
        print ("Error Mode! Exit!")
        sys.exit(0)


def load_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]
    num_paddings = hp.reduction_factor - (t % hp.reduction_factor) if t % hp.reduction_factor != 0 else 0 # for reduction
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel.reshape((-1, hp.num_mels * hp.reduction_factor)), mag
