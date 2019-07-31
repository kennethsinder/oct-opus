import os
import sys
import argparse
import glob

import tensorflow as tf

from src.train import train
from src.model_state import ModelState
from src.utils import generate_inferred_images
from src.parameters import GPU, BUFFER_SIZE, TEST_DATA_DIR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'], help='Specify the mode in which to run the mode')
    parser.add_argument('logdir', help='Specify where to store the Tensorboard logs')
    parser.add_argument('-r', '--restore', action='store_true', help='Restore model state from latest checkpoint')
    parser.add_argument('-s', '--step', type=int, required=True, help="The step that Tensorboard is currently")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

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
            step = train(model_state, args.step, writer)
            sys.exit(step)
    else:
        # load from latest checkpoint
        model_state.restore_from_checkpoint()

        # generate results based on prediction
        generate_inferred_images(model_state, TEST_DATA_DIR)
