#-*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.seq2seq import CustomHelper
import embedding
from embedding import prenet
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


