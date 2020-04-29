from cgan.parameters import USE_K_FOLDS

import argparse
import os
import time
import datetime

import tensorflow as tf

from cgan.parameters import GPU
from datasets.train_and_test import K
from cgan.model_state import ModelState
from cgan.train import train_epoch
from cgan.utils import generate_inferred_images, generate_cross_section_comparison

# This is why we can't have nice things:
# https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
# (Also, this doesn't seem to be affecting the verbosity much if at all...)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'], help='Specify the mode in which to run the program')
    parser.add_argument('hardware', choices=['cpu', 'gpu'], help='Specify whether script is being run on CPU or GPU')
    parser.add_argument('-l', '--logdir', metavar='PATH', help='Specify where to store the Tensorboard logs')
    parser.add_argument('-s', '--starting-epoch', type=int, help='Specify the initial epoch number', default=1)
    parser.add_argument('-e', '--ending-epoch', type=int, help='Specify the final epoch number', default=10)
    parser.add_argument('-d', '--datadir', help='Specify the root directory to look for data')
    return parser.parse_args()


if __name__ == '__main__':
    # main directory used to store output
    EXP_DIR = "experiment-{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    os.makedirs(EXP_DIR, exist_ok=False)

    args = get_args()

    if args.hardware == 'gpu':
        device_name = tf.test.gpu_device_name()
        if device_name != GPU:
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    if args.mode == 'train':
        model_state = ModelState(args.datadir)
        num_epochs = args.ending_epoch - args.starting_epoch + 1
        # go through each of K=5 folds, goes from 0 to 4 inclusive
        for fold_num in range(K if USE_K_FOLDS else 1):
            if USE_K_FOLDS:
                print('----- Starting fold number {} -----'.format(fold_num))
            model_state.reset_weights()
            model_state.get_datasets(fold_num)

            # main epoch loop
            for epoch_num in range(args.starting_epoch, args.ending_epoch + 1):
                print('----- Starting epoch number {} -----'.format(epoch_num))
                start = time.time()
                train_epoch(model_state.train_dataset, model_state, epoch_num + fold_num * num_epochs)
                model_state.save_checkpoint()
                print('Time taken for epoch {} is {} sec\n'.format(epoch_num, time.time() - start))

                # cross-section image logging
                for inp, tar in model_state.test_dataset.take(1):
                    generate_cross_section_comparison(model_state.generator, inp, tar,
                                                      epoch_num + fold_num * num_epochs)

        model_state.cleanup()   # Delete .h5 files for scrambled-weight models

    """ Prediction/Testing Code. This is run either independently or after training has completed. """

    # load from latest checkpoint and load data for just 1 of 5 folds
    model_state = ModelState(args.datadir)
    model_state.restore_from_checkpoint()

    # generate results based on prediction
    generate_inferred_images(EXP_DIR, model_state)
