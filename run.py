from configs.parameters import GPU, ALL_DATA_DIR
from src.utils import generate_inferred_images
from src.train import train
from src.model_state import ModelState
import tensorflow as tf
import argparse

# This is why we can't have nice things:
# https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
# (Also, this doesn't seem to be affecting the verbosity much if at all...)
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=[
                        'train', 'predict'], help='Specify the mode in which to run the mode')
    parser.add_argument('hardware', choices=[
                        'cpu', 'gpu'], help='Specify whether script is being run on CPU or GPU')
    parser.add_argument('-l', '--logdir', metavar='PATH',
                        help='Specify where to store the Tensorboard logs')
    parser.add_argument('-r', '--restore', action='store_true',
                        help='Restore model state from latest checkpoint')
    parser.add_argument('-e', '--epoch', type=int,
                        help='Specify the (initial) epoch number', default=1)
    parser.add_argument('-d', '--datadir',
                        help='Specify the root directory to look for data')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.hardware == "gpu":
        device_name = tf.test.gpu_device_name()
        if device_name != GPU:
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    model_state = ModelState(args.datadir)

    if args.mode == 'train':
        if args.restore:
            # load from latest checkpoint
            model_state.restore_from_checkpoint()
        train(model_state, args.epoch)
    else:
        # load from latest checkpoint
        model_state.restore_from_checkpoint()

        # generate results based on prediction
        generate_inferred_images(model_state, args.epoch)
