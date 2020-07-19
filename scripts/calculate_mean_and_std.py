import argparse
import math
import os
from os.path import join

import tensorflow as tf

import cnn.image as image
import cnn.utils as utils


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', required=True)
    parser.add_argument('-s', '--split', type=float, required=True)
    parser.add_argument('-sd', '--seed', type=int, required=True)
    args = parser.parse_args()

    data_names, _ = utils.separate_training_testing(
        args.data_dir,
        args.split,
        args.seed
    )
    utils.log('Training datasets are {}'.format(data_names))

    bscan_paths = utils.get_bscan_paths(args.data_dir, data_names)

    utils.log('Calculating training data mean')
    mean = utils.get_mean(bscan_paths)
    utils.log('Done calculating training data mean')

    utils.log('Calculating training data standard deviation')
    std = utils.get_standard_deviation(bscan_paths, mean)
    utils.log('Done calculating training data standard deviation')

    utils.log(
        'Final Results:\n' +
        'data dir={}\n'.format(args.data_dir) +
        'split={}\n'.format(args.split) +
        'seed={}\n'.format(args.seed) +
        'mean={}\n'.format(mean) +
        'std.={}'.format(std)
    )


if __name__ == '__main__':
    main()
