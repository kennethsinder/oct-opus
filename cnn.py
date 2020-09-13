# September 2020 - cnn.py
#
# This code is an implemention of the convolutional neural network (CNN) approach
# of Dr. Aaron Y. Lee and their University of Washington team:
#
#     Lee, C.S., Tyring, A.J., Wu, Y., et al.
#     “Generating Retinal Flow Maps from Structural Optical Coherence Tomography with Artificial Intelligence,”
#     Scientific Reports 9, 5694 (2019).
#
# The paper can be found at: https://doi.org/10.1038/s41598-019-42042-y
# Credit goes to them, and we also thank them for helping us implement the model from their paper
# and particularly for their patience helping us reproduce the smaller details of their architecture.

import argparse
import os
import traceback
from datetime import datetime
from os.path import join

import tensorflow as tf

import cnn.utils as utils
from cnn.model import CNN
from cnn.parameters import (
    AUGMENT_NORMALIZE,
    AUGMENT_CONTRAST,
    AUGMENT_FULL,
    IMAGE_DIM,
    GPU
)
from cnn.enface import generate_enface

DEFAULT_K_FOLDS = 5
DEFAULT_SELECTED_FOLD = 0
DEFAULT_BATCH_SIZE = 25 #50 #400
DEFAULT_NUM_SLICES = 2
DEFAULT_SEED = 42


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')

# we want the image tensors to have the shape (num_channels, img_height, img_width)
tf.keras.backend.set_image_data_format('channels_first')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'enface'], help='Program mode.')
    parser.add_argument('hardware', choices=['cpu', 'gpu'])
    parser.add_argument(
        'augment_level',
        choices=[AUGMENT_NORMALIZE, AUGMENT_CONTRAST, AUGMENT_FULL],
        help='Which level of data augmentation to use. '
             f'{AUGMENT_NORMALIZE}=normalize bscans using mean and std. '
             f'{AUGMENT_CONTRAST}=increase constrast of bscans and omags. '
             f'{AUGMENT_FULL}=perform same augmentations as the cGAN.'
    )
    parser.add_argument('-d', '--data-dir',
        required=True, help='Top level data directory.')
    parser.add_argument('-ef', '--enface-dir',
        help='For enface mode, the directory containing the input eye data.')
    parser.add_argument('-ex', '--experiment-dir',
        help='Directory in which the experiment data (e.g. checkpoints, enfaces) is saved.')
    parser.add_argument('-k', '--k-folds',
        type=int, default=DEFAULT_K_FOLDS, help='Number of folds to use when training.')
    parser.add_argument('-s', '--selected-fold',
        type=int, default=DEFAULT_SELECTED_FOLD, help='Which fold to train on. Index starts at 0.')
    parser.add_argument(
        '-sl', '--slices',
        type=int,
        default=DEFAULT_NUM_SLICES,
        help='Number of slices to cut the data into. '
             'Image width must be divisble by this number.'
    )
    parser.add_argument('-b', '--batch',
        type=int, default=DEFAULT_BATCH_SIZE, help='Training/testing batch size.')
    parser.add_argument('-sd', '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Seed used for Python\'s pseudo-random number generator. '
             'Affects how datasets are partitioned for the k-folds validation.'
    )
    parser.add_argument('-e', '--num-epochs',
        type=int, default=1, help='Number of epochs to train for.')
    parser.add_argument('-n', '--normalize', action='store_true',
        help='Set this flag to generated normalized enfaces.')
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
        args.k_folds,
        args.selected_fold,
        args.augment_level,
        args.batch,
        args.slices,
        args.seed,
        args.experiment_dir
    )

    if args.mode == 'train':
        if args.num_epochs < 1:
            raise Exception('Number of epochs must be at least one.')

        model.load_data()
        utils.log('training={}'.format(model.training_dirs))
        utils.log('testing={}'.format(model.testing_dirs))

        model.train(args.num_epochs)
        utils.log('Generating enfaces')
        for dir in model.testing_dirs:
            try:
                generate_enface(model, dir, normalize=args.normalize)
            except Exception:
                utils.log('Could not generate enfaces for {}, got the following exception:\n{}\nSkipping enfaces for {}'.format(
                    dir,
                    traceback.format_exc(),
                    dir
                ))
                with open(join(model.enfaces_dir, 'skipped.csv'), 'a') as file:
                    file.write('{}, {}\n'.format(model.epoch.numpy(), dir))
    else:
        if args.enface_dir is None:
            raise Exception('Enface directory must be specified.')
        generate_enface(model, args.enface_dir, normalize=args.normalize)


if __name__ == '__main__':
    main()
