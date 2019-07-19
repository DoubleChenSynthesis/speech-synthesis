#-*-coding:utf-8-*-
import argparse
import os
from tqdm import tqdm
from hyperparameters import hyperparameters
import thchs30

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapre',choices=['thchs30'])
    args = parser.parse_args()
    if args.dataset == 'thchs30':
        datapre_thc(args)
