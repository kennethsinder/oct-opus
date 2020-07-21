import argparse
import os
from datetime import datetime
from os.path import join

import tensorflow as tf

import cnn.utils as utils
from cnn.model import CNN
from cnn.parameters import IMAGE_DIM, GPU
from cnn.enface import generate_enface

DEFAULT_DATA_SPLIT = 0.8
DEFAULT_BATCH_SIZE = 25 #50 #400
DEFAULT_NUM_SLICES = 2
DEFAULT_SEED = 42


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')

# we want the image tensors to have the shape (num_channels, img_height, img_width)
tf.keras.backend.set_image_data_format('channels_first')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'enface'])
    parser.add_argument('hardware', choices=['cpu', 'gpu'])
    parser.add_argument('-d', '--data-dir', required=True)
    parser.add_argument('-ef', '--enface-dir')
    parser.add_argument('-ex', '--experiment-dir')
    parser.add_argument('-s', '--split', type=float, default=DEFAULT_DATA_SPLIT)
    parser.add_argument('-sl', '--slices', type=int, default=DEFAULT_NUM_SLICES)
    parser.add_argument('-b', '--batch', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('-sd', '--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('-e', '--num-epochs', type=int, default=1)
    args = parser.parse_args()

    if args.hardware == 'gpu':
        device_name = tf.test.gpu_device_name()
        if device_name != GPU:
            raise SystemError('GPU device not found')
        utils.log('Found GPU at: {}'.format(device_name))

    if IMAGE_DIM % args.slices != 0:
        raise Exception('Image width must be divisble by the number of slices')

    if args.experiment_dir is None:
        args.experiment_dir = join(
            './experiment', datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss'))

    model = CNN(
        args.data_dir,
        args.split,
        args.batch,
        args.slices,
        args.seed,
        args.experiment_dir
    )

    utils.log('training={}'.format(model.training_data_names))
    utils.log('testing={}'.format(model.testing_data_names))
    utils.log('mean={}, std={}'.format(model.training_mean, model.training_std))

    if args.mode == 'train':
        if args.num_epochs < 1:
            raise Exception('Number of epochs must be at least one.')
        utils.log('Saving experiment info in {}'.format(args.experiment_dir))
        model.train(args.num_epochs)
    else:
        if args.enface_dir is None:
            raise Exception('Enface directory must be specified.')
        generate_enface(model, args.enface_dir, verbose=True)


if __name__ == '__main__':
    main()
