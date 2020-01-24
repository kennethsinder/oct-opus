from configs.parameters import EXPERIMENT
assert EXPERIMENT.alive  # Needed to due import dependency issues

import argparse
# This is why we can't have nice things:
# https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
# (Also, this doesn't seem to be affecting the verbosity much if at all...)
import os
import time

import tensorflow as tf

from configs.parameters import GPU
from src.model_state import ModelState
from src.train import train_epoch
from src.utils import generate_inferred_images, generate_cross_section_comparison

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'], help='Specify the mode in which to run the mode')
    parser.add_argument('hardware', choices=['cpu', 'gpu'], help='Specify whether script is being run on CPU or GPU')
    parser.add_argument('-l', '--logdir', metavar='PATH', help='Specify where to store the Tensorboard logs')
    parser.add_argument('-r', '--restore', action='store_true', help='Restore model state from latest checkpoint')
    parser.add_argument('-s', '--starting-epoch', type=int, help='Specify the initial epoch number', default=1)
    parser.add_argument('-e', '--ending-epoch', type=int, help='Specify the final epoch number', default=10)
    parser.add_argument('-d', '--datadir', help='Specify the root directory to look for data')
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
        # main epoch loop
        for epoch_num in range(args.starting_epoch, args.ending_epoch):
            print('----- Starting epoch number {} -----'.format(epoch_num))
            start = time.time()
            train_epoch(model_state.train_dataset, model_state, epoch_num)
            model_state.save_checkpoint()
            print('Time taken for epoch {} is {} sec\n'.format(epoch_num, time.time() - start))

            # cross-section image logging
            for inp, tar in model_state.test_dataset.take(1):
                generate_cross_section_comparison(model_state.generator, inp, tar, epoch_num)

            # enface image logging
            if epoch_num % 5 == 0:
                generate_inferred_images(model_state, epoch_num)
                print('Generated inferred images for epoch {}'.format(epoch_num))

    else:
        # load from latest checkpoint
        model_state.restore_from_checkpoint()
        # generate results based on prediction
        generate_inferred_images(model_state, args.epoch)
