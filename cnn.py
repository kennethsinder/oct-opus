import argparse
from datetime import datetime
from os.path import join

import tensorflow as tf

import cnn.utils as utils
from cnn.model import CNN
from cnn.parameters import BATCH_SIZE, GPU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'enface'])
    parser.add_argument('hardware', choices=['cpu', 'gpu'])
    parser.add_argument('-d', '--data-dir', required=True, metavar='<path>')
    parser.add_argument('-s', '--split', type=float, default=0.8, metavar='<float>')
    parser.add_argument('-b', '--batch', type=int, default=BATCH_SIZE, metavar='<int>')
    parser.add_argument(
        '-c', '--checkpoints-dir',
        default=join('./checkpoints', datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')),
        metavar='<path>'
    )
    parser.add_argument('-e', '--num-epochs', type=int, default=1, metavar='<int>')
    parser.add_argument('-ef', '--enface-dir', metavar='<path>')
    args = parser.parse_args()

    if args.hardware == 'gpu':
        device_name = tf.test.gpu_device_name()
        if device_name != GPU:
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    model = CNN(args.data_dir, args.split, args.batch, args.checkpoints_dir)

    if args.mode == 'train':
        print('Saving checkpoints to {}'.format(model.checkpoints_dir))
        if args.num_epochs < 1:
            raise('Number of epochs must be at least one.')
        model.train(args.num_epochs)
    else:
        utils.generate_enface(model, args.enface_dir)


if __name__ == '__main__':
    main()
