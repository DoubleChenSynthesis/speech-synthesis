#-*-coding:utf-8-*-
import tensorflow as tf
from embedding import *
import utils

#tacotron中的主结构
def cbhg(inputs,k,highway_layers,training,scope = 'cbhg'):
    with tf.variable_scope(scope):
        outputs = conv1d_banks(inputs = inputs,k = k,filters = 128,training=training)
        outputs = tf.layers.max_pooling1(inputs = outputs,pool_size = 2,strides = 1,padding = 'same')
        outputs = conv1d(inputs = outputs,kernel_size=3,filters=128,scope='conv1d1')
        outputs = batch_normalization(outputs,training,scope='batch_normalization1')
        outputs = tf.nn.relu(outputs)
        outputs = conv1d(inputs = outputs,kernel_size=3,filters=128,scope='conv1d2')
        outputs = batch_normalization(outputs,training,scope='batch_normalization2')
        outputs = outputs + inputs
        for i in range(highway_layers):
            outputs = highway(inputs = outputs,nodes=128,scope='highway_{}'.format(i))
        outputs = gru(inputs = outputs,nodes=128,bidrection=True)
    return outputs

#try paper
def transcript_encoder(inputs,embedding_size,k,highway_layers,training,scope='transcript_encoder'):
    #[batch_size,text_length]->[batch_size,text_length,256]
    with tf.variable_scope(scope):
        character_size = inputs.get_shape().as_list()[-1]
        embedding = embeding(inputs,character_size,embedding_size,scope='transcript_encoder')
        prenet_out = prenet(embedding,256,128,training)
        text = cbhg(inputs = prenet_out,k = k,highway_layers= highway_layers,training = training)
    return text

def reference_encoder(inputs,training,scope = 'reference_encoder'):
    #韵律向量？？？ 6 layers conv2d
    with tf.variable_scope(scope):
        inputs = tf.layers.conv2d(inputs = inputs,filters = 32,kernel_size=3,strides = 2,padding = 'SAME')
        inputs = batch_normalization(inputs,training,scope='batch_normalization1')
        inputs = tf.nn.relu(inputs)
        inputs = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=3, strides=2, padding='SAME')
        inputs = batch_normalization(inputs, training, scope='batch_normalization2')
        inputs = tf.nn.relu(inputs)
        inputs = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=3, strides=2, padding='SAME')
        inputs = batch_normalization(inputs, training, scope='batch_normalization3')
        inputs = tf.nn.relu(inputs)
        inputs = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=3, strides=2, padding='SAME')
        inputs = batch_normalization(inputs, training, scope='batch_normalization4')
        inputs = tf.nn.relu(inputs)
        inputs = tf.layers.conv2d(inputs=inputs, filters=128, kernel_size=3, strides=2, padding='SAME')
        inputs = batch_normalization(inputs, training, scope='batch_normalization5')
        inputs = tf.nn.relu(inputs)
        inputs = tf.layers.conv2d(inputs=inputs, filters=128, kernel_size=3, strides=2, padding='SAME')
        inputs = batch_normalization(inputs, training, scope='batch_normalization6')
        inputs = tf.nn.relu(inputs)

        ##inputs-->[batch,_,num_mel/64,128]

        bat,_,nm,yi = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs,[bat,-1,nm*yi])
        ##inputs-->[batch,_,num_mel/64*128]

        inputs = gru(inputs = inputs,nodes= 128)

        ##inputs-->[batch,_,128]
        inputs = inputs[:,-1,:]

        ##inputs-->[batch,128]
        prosody = tf.layers.dense(inputs = inputs,units = 128,activation=tf.nn.tanh)

    return prosody

def attention_gru_decoder(inputs,inputs_length,memory,attention_rnn_nodes,decoder_rnn_nodes,num_mels,reduction_factor,max_iters,training,scope = 'attention_gru_decoder'):
    #[batch,Ty/r,n_mel]-->[batch,T,D]
    with tf.variable_scope(scope):
        batch_size = memory.get_shape().as_list()[0]
        bahdanau_attention = tf.contrib.seq2seq.BahdanauAttention(num_units = attention_rnn_nodes,memory = memory)
        decoder_cell = tf.contrib.rnn.GRUCell(attention_rnn_nodes)

        cell_attention = tf.contrib.seq2seq.AttentionWrapper(cell = utils.decoder_prenet_wrapper(decoder_cell,training),attention_mechanism = bahdanau_attention, alignment_history = True,output_attention = False)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.OutputProjectionWrapper(cell_attention,attention_rnn_nodes),tf.contrib.rnn.ResidualWrapper(cell = tf.contrib.rnn.GRUCell(decoder_rnn_nodes)),tf.contrib.rnn.ResidualWrapper(cell = tf.contrib.rnn.GRUCell(decoder_rnn_nodes))],state_is_tuple = True)

        output_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell,num_mels*reduction_factor)
        decoder_init_state = output_cell.zero_state(batch_size=batch_size,dtype=tf.float32)
        #[batch,Ty/r,n_mel*r]
        if training:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs = inputs,sequence_length = tf.cast(tf.reshape(inputs_length,[batch_size]),tf.int32),time_major = False,name = 'training_helper')

        else:
            helper = utils.inference_helper(batch_size = batch_size,out_size=num_mels*reduction_factor)

        decoder = tf.contrib.seq2seq.BasicDecoder(cell = output_cell,helper = helper,initial_state = decoder_init_state)
        decoder_outputs,_,final_decoder_states,_ = tf.contrib.seq2seq.dynamic_decode(decoder = decoder,impute_finished = True,maximum_iterations = max_iters)

        alignments = tf.transpose(final_decoder_states[0].alignment_history.stack(),[1,2,0])
        mel_hats = tf.identity(decoder_outputs,name='mel_hats')

    return mel_hats,alignments

def cbhg_postprocessing(inputs,num_mels,num_fft,k,highway_layers,training,scope ='cbhg_postprocessing'):
    with tf.variable_scope(scope):
        #inputs-->[batch,Ty,num_mels]
        inputs = tf.reshape(inputs,[inputs.get_shape().as_list()[0],-1,num_mels])

        outputs = conv1d_banks(inputs=inputs,k=k,filters=128,training=training)
        outputs = tf.layers.max_pooling1d(inputs = outputs,pool_size=2,strides=1,padding='SAME')
        outputs = conv1d(inputs=outputs,kernel_size=3,filters=256,scope='conv1d1')
        outputs = batch_normalization(outputs,training,scope='batch_normalization1')
        outputs = tf.nn.relu(outputs)
        outputs = conv1d(inputs=outputs,kernel_size=3,filters=80,scope='conv1d2')
        outputs = batch_normalization(outputs,training,scope='batch_normalization2')
        outputs = outputs + inputs
        outputs = tf.layers.dense(outputs,128)
        for i in range(highway_layers):
            outputs = highway(inputs = outputs, nodes = 128, scope = "highway_{}".format(i))
        outputs = gru(inputs = outputs,nodes=128,bidrection=True)
        outputs = tf.layers.dense(outputs,1+num_fft//2)
        ##[batch,Ty,1+n_fft//2]
        return outputs




















