#-*-coding:utf-8-*-
from __future__ import division,print_function
import os
import re
import unicodedata
import codecs
import numpy as np
import tensorflow as tf
from hyperparameters import hyperparameters as hp

def load_vocabulary():
    char_idx = {char:idx for idx,char in enumerate(hp.characs)}
    idx_char = {idx:char for idx,char in enumerate(hp.characs)}
    return char_idx,idx_char

#处理文字
def text_normalization(text):
    text = ' '.join(char for char in unicodedata.normalize('NFD',text) if unicodedata.category(char) != 'Mn')
    text = re.sub(u'[^{}]'.format(hp.characs)," ",text)
    text = re.sub('[ ]+',' ',text)
    return  text

def parser(transcript,ref_path):
    #str-->int
    transcript = tf.decode_raw(transcript,tf.int32)
    transcript = tf.pad(transcript,([0,hp.num_charac],))[:hp.num_charac]
    def load_audio(audio_path):
        sub_paths = audio_path.strip().split('/')
        main_path = '/'.join(sub_paths[:-2])
        fname = os.path.basename(audio_path)
        mel_path = main_path + "/mels/{}".format(sub_paths[-1].replace('wav', 'npy'))
        mag_path = main_path + "/mags/{}".format(sub_paths[-1].replace('wav', 'npy'))
        mel = np.load(mel_path)
        mag = np.load(mag_path)
        ref_len = mel.shape[0] if mel.shape[1] == hp.num_mels * hp.reduction_factor \
            else mel.shape[0] * mel.shape[1] // hp.num_mels * hp.reduction_factor
        ref_len = np.array(ref_len, dtype=np.int32)
        return mel,mag,ref_len
    spectrogram,waveform,ref_len = tf.py_func(load_audio,[ref_path],[tf.float32,tf.float32,tf.int32])
    transcript = tf.shape(transcript,[hp.num_charac,])
    spectrogram = tf.reshape(spectrogram,[-1,hp.num_mels*hp.reduction_factor])
    waveform = tf.reshape(waveform,[-1,hp.num_fft//2+1])
    inputs = list((transcript,spectrogram,tf.reshape(ref_len,[1]),tf.reshape(tf.constant(0),[1]),spectrogram,waveform))
    return inputs

def input_fn(mode):
    data_dir = hp.train_data_dir if mode == 'train' else hp.eval_data_dir
    batch_size = hp.train_batch_size if mode == 'train' else hp.eval_batch_size
    char_idx,idx_char = load_vocabulary()
    lines = codecs.open(os.path.join(data_dir,'metadata.csv'),'r','utf-8').readlines()
    transcripts = []
    reference_paths = []
    for line in lines:
        fname,_,transcript = line.strip().split('|')
        reference_path = os.path.join(data_dir,'wavs',fname+'.wav')
        reference_paths.append(reference_path)
        transcript = text_normalization(transcript)+u"␃"
        transcript = [char_idx[char] for char in transcript]
        transcripts.append(np.array(transcript,np.int32).tostring())

    transcripts = tf.convert_to_tensor(transcripts)
    reference_paths = tf.convert_to_tensor(reference_paths)

    dataset = tf.data.Dataset.from_tensor_slices((transcripts,reference_paths))
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=100000)
        dataset = dataset.repeat()
    dataset = dataset.map(parser,num_parallel_calls=hp.data_num_parallel_calls)
    dataset = dataset.padded_batch(batch_size=batch_size,padded_shapes=([hp.num_charac],[None,hp.num_mels*hp.reduction_factor],[1],[1],[None,hp.num_mels*hp.reduction_factor],[None,hp.num_fft//2+1]))
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    batch_inputs = iterator.get_next()
    names = ['transcript','reference','ref_len','speaker','decoder','labels']
    batch_inputs = {name:inp for name,inp in zip(names,batch_inputs)}
    return batch_inputs