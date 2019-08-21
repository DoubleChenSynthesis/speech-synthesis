#-*-coding:utf-8-*-
from __future__ import division,print_function
import os
from scipy.io.wavfile import write
import tensorflow as tf
import data
import eval
from all_model import Model
from hyperparameters import hyperparameters as hp
import signal_process
def train_per_eval(session_config):
    with tf.Session(config=session_config)as sess:
        batch_inputs = data.input_fn(mode='train')
        model = Model(mode='train',inputs = batch_inputs)
        global_step = 0
        saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(hp.logdir,graph=sess.graph)
        if hp.train_from_restore:
            latest_model = tf.train.latest_checkpoint(os.path.join(hp.logdir,'models'))
            if hp.restore_model is not None and hp.restore_model != latest_model:
                restore_model = hp.restore_model
            else:
                restore_model = latest_model
            saver.restore(sess,restore_model)
        while global_step % hp.train_step_per_eval != 0 or global_step==0:
            out = sess.run([model.train_op,model.global_step,model.alignments,model.mag_hat,model.merged_labels,model.merged])
            global_step = out[1]
            summary = out[-1]
            summary_writer.add_summary(summary,global_step)
            if global_step%hp.save_model_step==0 and global_step!=0:
                save_model_dir = os.path.join(hp.logdir,'models')
                if not os.path.exists(save_model_dir):
                    os.mkdir(save_model_dir)
                saver.save(sess,os.path.join(save_model_dir,'model'),global_step=global_step)
                hp.train_from_restore=True
                save_sample_dir = os.path.join(hp.logdir,'train')
                if not os.path.exists(save_sample_dir):
                    os.mkdir(save_sample_dir)
                wav_hat=signal_process.spectrogrom2wav(out[3][0])
                ground_truth=signal_process.spectrogrom2wav(out[4][0])
                signal_process.plot_alignment(out[2][0],gs=global_step,mode='save_fig',path=save_sample_dir)
                write(os.path.join(save_sample_dir, 'gt_{}.wav'.format(global_step)), hp.sample_rate, ground_truth)
                write(os.path.join(save_sample_dir, 'hat_{}.wav'.format(global_step)), hp.sample_rate, wav_hat)
                merged = sess.run(tf.summary.merge([tf.summary.audio("train/sample_gt" + str(global_step), tf.expand_dims(ground_truth, 0),hp.sample_rate),
                     tf.summary.audio("train/sample_hat_gs" + str(global_step), tf.expand_dims(wav_hat, 0),hp.sample_rate),
                     tf.summary.image("train/attention_gs" + str(global_step),signal_process.plot_alignment(out[2][0], gs=global_step, mode="with_return"))]))
                summary_writer.add_summary(merged, global_step)

def train(session_config):
    for _ in range(hp.train_step//hp.train_step_per_eval):
        train_per_eval(session_config)
        tf.reset_default_graph()
        eval.eval(session_config)
        tf.reset_default_graph()