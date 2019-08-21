#-*-coding:utf-8-*-
import argparse
import os,sys
from tqdm import tqdm
from hyperparameters import hyperparameters
import thchs30
import train
import eval
import synthesize
import tensorflow as tf

def write_media(media,output):
    with open(os.path.join(output,'train.txt'),'w',encoding='utf-8') as f:
        for i in media:
            f.write('|'.join([str(x)] for x in i) +'\n')

    frames = sum([i[2] for i in media])
    hours = frames * hyperparameters.frame_shift_ms/(3600 *1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(media), frames, hours))
    print('Max input length:  %d' % max(len(i[3]) for i in media))
    print('Max output length: %d' % max(i[2] for i in media))



def datapre_thc(args):
    input = os.path.join(args.base_dir,'data_thchs30')
    output = os.path.join(args.base_dir, args.output)
    os.makedirs(output,exist_ok=True)
    media = thchs30.build_from_path(input,output,args.num_workers,tqdm=tqdm)
    write_media(media,output)


def main(argv):
    # Set Enviroment and GPU Options
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.INFO)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=hyperparameters.inter_op_parallelism_threads,
        intra_op_parallelism_threads=hyperparameters.intra_op_parallelism_threads,
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=gpu_options)
    session_config.gpu_options.allow_growth = True

    # Set log dir specifically
    hyperparameters.logdir = os.path.join(hyperparameters.logdir, "test{}".format(sys.argv[1]))

    if sys.argv[2] == 'train':
        # Train branch (Train branch also contains Eval branch, see train.py and Hyperparameter.py for more details)
        print("Training Mode")
        train.train(session_config)

    elif sys.argv[2] == 'eval':
        # Eval
        print("Evaluation Mode")
        eval.eval(session_config)

    elif sys.argv[2] == 'synthes':
        print("Synthesize Mode")
        synthesize.synthesize(session_config)

    else:
        print("Uncognized mode! You need type mode chosen from train/eval/synthes.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('python main.py num mode\nExample: python main.py 1 train/eval/synthes')

    main(sys.argv)
