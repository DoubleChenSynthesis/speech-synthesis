#-*-coding:utf-8-*-
from __future__ import division,print_function

import tensorflow as tf

import  model
from hyperparameters import  hyperparameters as hp

class Model(object):
    def __init__(self,mode,inputs = None):
        with tf.device('/cpu:0'):
            self.mode = mode
            self.inputs_transcript = [None]*hp.num_gpus
            self.inputs_reference = [None]*hp.num_gpus
            self.inputs_reference_length = [None]*hp.num_gpus
            self.inputs_speaker = [None]*hp.num_gpus
            self.inputs_decoder = [None]*hp.num_gpus
            self.labels = [None]*hp.num_gpus
            if inputs is not None:
                self.max_batch_length = tf.cast(tf.reduce_max(inputs['reference_length']),dtype=tf.int32)
                self.feed_gpu(inputs)
            else:
                self.max_batch_length = hp.max_batch_length
            self.tower_memory = [None]*hp.num_gpus
            self.tower_mel_hat = [None]*hp.num_gpus
            self.tower_alignments = [None]*hp.num_gpus
            self.tower_mag_hat = [None]*hp.num_gpus

            for gpu_id in range(hp.num_gpus):
                with tf.device('gpu:%d'%gpu_id):
                    with tf.name_scope('tower_%d'%gpu_id):
                        with tf.variable_scope('cpu_variables',reuse=gpu_id>0):
                            if mode == 'synthesize':
                                #[batch Tx]
                                self.inputs_transcript[gpu_id] = tf.placeholder(tf.int32,shape=[None,hp.num_charac],name='inputs_transcript')
                                #[batch,Ty/r,n_mels*r]
                                self.inputs_reference[gpu_id] = tf.placeholder(tf.float32,shape=[None,None,hp.num_mels * hp.reduction_factor],name='inputs_reference')
                                #[batch,1]
                                self.inputs_reference_length[gpu_id] = tf.placeholder(tf.int32,shape=[None,1],name='inputs_reference_length')
                                #[batch,1]
                                self.inputs_speaker[gpu_id] = tf.placeholder(tf.int32,shape=[None,1],name='inputs_speaker')
                                #[batch,Ty/r,n_mels*r]
                                self.inputs_decoder[gpu_id] = tf.placeholder(tf.float32,shape=[None,None,hp.num_mels*hp.reduction_factor],name='inputs_decoder')
                                #[batch,
                                self.labels[gpu_id] = tf.placeholder(tf.float32,shape=[None,None,hp.num_fft//2+1],name='inputs_label')

                            mel_hat,alignments,mag_hat = self.single_model(gpu_id)
                            self.tower_mel_hat[gpu_id] = mel_hat
                            self.tower_alignments[gpu_id] = alignments
                            self.tower_mag_hat[gpu_id] = mag_hat

            if self.mode == 'train' or self.mode == 'eval':
                self.merged_labels = tf.concat(self.labels,0)
            self.mel_hat = tf.concat(self.tower_mag_hat,0)
            self.alignments = tf.concat(self.tower_alignments,0)
            self.mag_hat = tf.concat(self.tower_mag_hat,0)
            if self.mode == 'train':
                self.add_loss_op()
                self.add_train_op()
            elif self.mode == 'eval':
                self.add_loss_op()

            self.add_summary_op()
            self.merged = tf.summary.merge_all()

    def single_model(self,gpu_id):
        inputs_transcript = self.inputs_transcript[gpu_id]
        inputs_reference = self.inputs_reference[gpu_id]
        inputs_reference_length = self.inputs_reference_length[gpu_id]
        inputs_speaker = self.inputs_speaker[gpu_id]
        inputs_decoder = self.inputs_decoder[gpu_id]

        if self.mode == 'train':
            training = True
        else:
            training = False

        #transcript encoder [batch,text_length,256]
        text = model.transcript_encoder(inputs_transcript,embedding_size=hp.charac_embed_size,k = hp.num_encoder_banks,highway_layers=hp.num_enc_highway_layers,training=training)
        text = tf.identity(text,name='text_encoder')

        #reference encoder
        if self.mode == 'train':
            batch_size = hp.train_batch_size//hp.num_gpus
        elif self.mode == 'eval':
            batch_size = hp.eval_batch_size//hp.num_gpus
        else:
            batch_size = hp.synthes_batch_size//hp.num_gpus
        inputs_reference_reshape = tf.reshape(inputs_reference,[batch_size,-1,hp.num_mels])
        #[batch,Ty,n_mels]-->[batch,Ty,n_mels,1]
        inputs_reference_reshape = tf.expand_dims(inputs_reference_reshape,-1)
        prosody = model.reference_encoder(inputs=inputs_reference_reshape,training=training)#[batch,128]
        prosody = tf.expand_dims(prosody,1)#[batch,128]
        prosody = tf.tile(prosody,[1,hp.num_charac,1],name='prosody_encoder')#[batch,Tx,128]
        if hp.num_speakers > 1:
            #[batch,1,speaker_embed_size][32,1,16]
            speaker = model.embeding(inputs = inputs_speaker,character_size=hp.num_speakers,embedding_size=hp.speaker_embed_size,scope='speaker')
            #[batch,num_charac,speaker_embed_size][32,1,16]
            speaker = tf.tile(speaker,[1,hp.num_charac,1],name='speaker_embedding')
            #[batch,Tx,Dt+Ds+Dp]
            memory = tf.concat([text,prosody,speaker],axis=-1,name='memory')
        else:
            # [batch,Tx,Dt+Ds]
            memory = tf.concat([text, prosody], axis=-1, name='memory')

        #spectrogrom decoder
        if self.mode == 'train':
            #[batch,Ty/r,num_mels*r]
            inputs_decoder = tf.concat(tf.zeros_like(inputs_decoder[:,:1,:],inputs_decoder[:,:-1,:]),1)
        #[batch,Ty/r,num_mels*r]
        mel_hat,alignments = model.attention_gru_decoder(inputs=inputs_decoder,inputs_length=inputs_reference_length,memory=memory,attention_rnn_nodes=hp.num_attention_nodes,decoder_rnn_nodes=hp.num_decoder_nodes,num_mels=hp.num_mels,reduction_factor=hp.reduction_factor,max_iters=self.max_batch_length,training=training)
        alignments = tf.identity(alignments,name='alignments')
        mel_hat = tf.identity(mel_hat,name='mel_hat')
        #[batch,Ty,1+n_fft/2]
        mag_hat = model.cbhg_postprocessing(inputs=mel_hat,num_mels=hp.num_mels,num_fft=hp.num_fft,k=hp.num_post_banks,highway_layers=hp.num_post_highway_layers,training=training)
        mag_hat = tf.identity(mag_hat,name='mag_hat')
        return mel_hat,alignments,mag_hat

    def add_loss_op(self):
        self.tower_loss = []
        self.tower_loss1 = []
        self.tower_loss2 = []
        if self.mode == 'train':
            batch_size = hp.train_batch_size//hp.num_gpus
        elif self.mode == 'eval':
            batch_size = hp.eval_batch_size//hp.num_gpus

        def calculator_loss(gpu_id):
            with tf.device('gpu:%d'%gpu_id):
                with tf.name_scope('tower_%d'%gpu_id):
                    with tf.variable_scope('cpu_variables',reuse=gpu_id>0):
                        loss1 = tf.reduce_sum(tf.abs(self.tower_mel_hat[gpu_id]-self.inputs_reference[gpu_id]))
                        loss2 = tf.reduce_sum(tf.abs(self.tower_mag_hat[gpu_id]-self.labels[gpu_id]))
                        return loss1,loss2
        for i in range(hp.num_gpus):
            loss1,loss2 = calculator_loss(i)
            self.tower_loss1.append(loss1/batch_size)
            self.tower_loss2.append(loss2/batch_size)
            self.tower_loss.append((loss1+loss2)/batch_size)
        self.loss1 = tf.reduce_mean(self.tower_loss1,name='seq2seq_loss')
        self.loss2 = tf.reduce_mean(self.tower_loss2,name='output_loss')
        self.loss = tf.reduce_mean(self.tower_loss,name='total_loss')

    def add_train_op(self):
        self.global_step = tf.Variable(0,name='global_step',trainable=False)
        def learning_rate_decay(init_lr,global_step,warmup_steps=4000.):
            step = tf.cast(global_step+1,dtype=tf.float32)
            return init_lr * warmup_steps **0.5*tf.minimum(step*warmup_steps**-1.5,step**-0.5)
        self.learning_rate = learning_rate_decay(hp.learning_rate,self.global_step)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.tower_grads=[]
        def calculator_grad(gpu_id):
            with tf.device('gpu:%d'%gpu_id):
                with tf.name_scope('tower_%d'%gpu_id):
                    with tf.variable_scope('cpu_variables',reuse=gpu_id>0):
                        gvs = self.optimizer.compute_gradients(self.tower_loss[gpu_id])
                        return gvs
        for i in range(hp.num_gpus):
            self.tower_grads.append(calculator_grad(i))
        def average_grad(tower_grad):
            average_grad=[]
            for i in zip(*tower_grad):
                grads = [j for j,_ in i]
                grad = tf.stack(grads,0)
                grad = tf.reduce_mean(grad,0)
                v = i[0][1]
                grad_var = (grad,v)
                average_grad.append(grad_var)
            return average_grad
        self.grad = average_grad(self.tower_grads)
        def grad_clipping(gvs):
            clip = []
            for grad,var in gvs:
                grad = tf.clip_by_norm(grad,5.)
                clip.append((grad,var))
                print(var)
            return clip
        self.grad = grad_clipping(self.grad)
        minimize_op = self.optimizer.apply_gradients(self.grad,global_step=self.global_step)
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group(minimize_op,update_op)

    def add_summary_op(self):
        if self.mode =='train' or self.mode =='eval':
            tf.summary.scalar('{}/seq2seq_loss'.format(self.mode), self.loss1)
            tf.summary.scalar('{}/output_loss'.format(self.mode), self.loss2)
            tf.summary.scalar('{}/total_loss'.format(self.mode), self.loss)
            if self.mode == 'train':
                tf.summary.scalar('{}/learning_rate'.format(self.mode), self.learning_rate)

    def feed_gpu(self,inputs):
        batch_size = hp.train_batch_size if self.mode == 'train' else hp.eval_batch_size
        batch_gpu = batch_size//hp.num_gpus
        for i in range(hp.num_gpus):
            self.inputs_transcript[i] = tf.slice(inputs['transcript'],[i*batch_gpu,0],[batch_gpu,-1])
            self.inputs_reference[i] = tf.slice(inputs['reference'],[i*batch_gpu,0,0],[batch_gpu,-1,-1])
            self.inputs_reference_length[i] = tf.slice(inputs['reference_length'],[i*batch_gpu,0],[batch_gpu,-1])
            self.inputs_speaker[i] = tf.slice(inputs['speaker'],[i*batch_gpu,0],[batch_gpu,-1])
            self.labels[i] = tf.slice(inputs['labels'],[i*batch_gpu,0,0],[batch_gpu,-1,-1])
            if self.mode == 'train':
                self.inputs_decoder[i] = tf.slice(inputs['decoder'],[i*batch_gpu,0,0],[batch_gpu,-1,-1])
