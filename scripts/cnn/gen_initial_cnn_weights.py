import argparse
import os

import tensorflow as tf

import cnn.utils as utils
from cnn.model import CNN
from cnn.parameters import IMAGE_DIM, GPU

DEFAULT_K_FOLDS = 5
DEFAULT_SELECTED_FOLD = 0
DEFAULT_BATCH_SIZE = 25 #50 #400
# DEFAULT_NUM_SLICES = 2
DEFAULT_SEED = 42
DEFAULT_CONTRAST = 1.0


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')

tf.keras.backend.set_image_data_format('channels_first')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hardware', choices=['cpu', 'gpu'])
    parser.add_argument('-d', '--data-dir', required=True)
    parser.add_argument('-ex', '--experiment-dir', required=True)
    parser.add_argument('-sl', '--slices', type=int, required=True)
    args = parser.parse_args()

    if args.hardware == 'gpu':
        device_name = tf.test.gpu_device_name()
        if device_name != GPU:
            raise SystemError('GPU device not found')
        utils.log('Found GPU at: {}'.format(device_name))

    if IMAGE_DIM % args.slices != 0:
        raise Exception('Image width must be divisble by the number of slices')

    model = CNN(
        args.data_dir,
        DEFAULT_K_FOLDS,
        DEFAULT_SELECTED_FOLD,
        DEFAULT_BATCH_SIZE,
        args.slices,
        DEFAULT_CONTRAST,
        DEFAULT_SEED,
        args.experiment_dir
    )

    utils.log('Saving initial checkpoint')
    model.save_checkpoint()

if __name__ == '__main__':
    main()
