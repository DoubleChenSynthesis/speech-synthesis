#-*-coding:utf-8-*-

import tensorflow as tf

class EmbeddingLayer(tf.layers.Layer):
    def __init__(self,text_size,hidden_size,scope):
        super(EmbeddingLayer,self).__init__()
        self.text_size = text_size
        self.hidden_size = hidden_size
        self.scope = scope

    def build(self):
        #创建和输入
        with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
            self.lookup_table = tf.get_variable("table_weights",[self.text_size,self.hidden_size],initializer=tf.random_normal_initializer(0.,self.hidden_size**-0.5))
            self.built = True

    def call(self,x):
        with tf.name_scope(self.scope):
            embeddings = tf.gather(self.lookup_table,x)
            embeddings *= self.hidden_size**0.5
            return embeddings

def embedding(inputs,character_size,embedding_size,scope="embedding"):
    #将character embedding
    embedding_layer = EmbeddingLayer(character_size,embedding_size,scope)
    return embedding_layer(inputs)

#批处理
def batch_normalization(inputs,training,data_format = "channels_last",scope = "bn"):
    with tf.variable_scope(scope):
        return tf.layers.batch_normalization(
            inputs = inputs,axis= 1 if data_format == "channels_first" else len(inputs.get_shape().as_list())-1,momentum = 0.997, epsilon = 1e-5,center = True, scale = True, training = training, fused = True
        )



#layers
#--------------------------------------------------------------------
def conv1d(inputs,kernel_size = 1,filters = 128, padding = 'SAME',dilation_rate = 1,data_format = "channels_last", scope = "conv1d"):
    with tf.variable_scope(scope):
        return tf.layers.conv1d(inputs = inputs,filters = filters,kernel_size = kernel_size,padding=padding,data_format=data_format,dilation_rate=dilation_rate)


def conv1d_banks(inputs,k,filters,training,scope = 'conv1d_banks'):
    #[batch,T,C]-->[batch,T,K*filters]
    with tf.variable_scope(scope):
        outputs = tf.concat(
            [conv1d(inputs,k,filters)]
        )

        return tf.nn.relu(batch_normalization(outputs,training))

def gru(inputs,nodes,bidrection = False, scope = 'gru'):
    #[N,TimeStep,D]-->[N,TimeStep,nodes]or[N,TimeStep,2*nodes]
    with tf.variable_scope(scope):
        if bidrection:
            gru_fw_cell = tf.contrib.rnn.GRUCell(nodes)
            gru_bw_cell = tf.contrib.rnn.GRUCell(nodes)
            outputs,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw = gru_fw_cell,cell_bw = gru_bw_cell,inputs = inputs,dtype = tf.float32)
            outputs = tf.concat(outputs,axis=2)

        else:
            cell = tf.contrib.rnn.GRUCell(nodes)
            outputs,_ = tf.nn.dynamic_rnn(cell = cell,inputs = inputs,dtype = tf.float32)

        return outputs

def prenet(inputs,nodes_dense1,nodes_dense2,training,scope = 'prenet'):
    with tf.variable_scoper(scope):
        outputs = tf.layers.dense(inputs = inputs,units=nodes_dense1,activation=tf.nn.relu,name = 'dense1')
        outputs = tf.layers.dropout(inputs = outputs,rate = 0.5,training = training,name = 'dropout1')
        outputs = tf.layers.dense(inputs = outputs,units = nodes_dense2,activation = tf.nn.relu,name = 'dense2')
        outputs = tf.layers.dropout(inputs = outputs,rate = 0.5,training = training,name='dropout2')
        return outputs

def highway(inputs,nodes,scope = 'highway'):
    with tf.variable_scope(scope):
        H = tf.layers.dense(inputs = inputs,units=nodes,activation=tf.nn.relu,name='hidden')
        T = tf.layers.dense(inputs = inputs,units=nodes,activation=tf.nn.sigmoid,bias_initializer=tf.constant_initializer(-1.0),name='transfrom_gate')
        return H * T + inputs *(1-T)
