import argparse
import os
from datetime import datetime
from os.path import join

import tensorflow as tf

from cnn.model import CNN
from cnn.parameters import BATCH_SIZE, DATA_SPLIT, GPU, SEED
from cnn.enface import generate_enface


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'enface'])
    parser.add_argument('hardware', choices=['cpu', 'gpu'])
    parser.add_argument('-d', '--data-dir', required=True, metavar='<path>')
    parser.add_argument('-ef', '--enface-dir', metavar='<path>')
    parser.add_argument('-ex', '--experiment-dir',
        default=join('./experiment', datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')),
        metavar='<path>')
    parser.add_argument('-s', '--split', type=float, default=DATA_SPLIT, metavar='<float>')
    parser.add_argument('-b', '--batch', type=int, default=BATCH_SIZE, metavar='<int>')
    parser.add_argument('-sd', '--seed', type=int, default=SEED, metavar='<int>')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, metavar='<int>')
    args = parser.parse_args()

    if args.hardware == 'gpu':
        device_name = tf.test.gpu_device_name()
        if device_name != GPU:
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    model = CNN(args.data_dir, args.split, args.batch, args.seed, args.experiment_dir)

    if args.mode == 'train':
        if args.num_epochs < 1:
            raise('Number of epochs must be at least one.')
        print('Saving experiment info in {}'.format(args.experiment_dir))
        model.train(args.num_epochs)
    else:
        if args.enface_dir is None:
            raise('Enface directory must be specified.')
        generate_enface(model, args.enface_dir, verbose=True)


if __name__ == '__main__':
    main()
