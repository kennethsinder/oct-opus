import argparse

import tensorflow as tf

from src.model_state import ModelState
from src.parameters import GPU, TEST_DATA_DIR
from src.train import train
from src.utils import generate_inferred_images


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'], help='Specify the mode in which to run the mode')
    parser.add_argument('hardware', choices=['cpu', 'gpu'], help='Specify whether script is being run on CPU or GPU')
    parser.add_argument('-l', '--logdir', metavar='PATH', help='Specify where to store the Tensorboard logs')
    parser.add_argument('-r', '--restore', action='store_true', help='Restore model state from latest checkpoint')
    parser.add_argument('-e', '--epoch', type=int, help='Specify the epoch number')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.hardware == "gpu":
        device_name = tf.test.gpu_device_name()
        if device_name != GPU:
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    model_state = ModelState()

    if args.mode == 'train':
        writer = tf.summary.create_file_writer(args.logdir)
        if args.restore:
            # load from latest checkpoint
            model_state.restore_from_checkpoint()
        train(model_state, writer, args.epoch)
    else:
        # load from latest checkpoint
        model_state.restore_from_checkpoint()

        # generate results based on prediction
        generate_inferred_images(model_state, TEST_DATA_DIR)
