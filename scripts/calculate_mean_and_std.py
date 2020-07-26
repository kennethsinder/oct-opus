import argparse
import math
import os
from os.path import join

import tensorflow as tf

import cnn.image as image
import cnn.utils as utils
from cnn.dataset import Dataset


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', required=True)
    parser.add_argument('-k', '--k-folds', type=int, required=True)
    parser.add_argument('-s', '--selected-fold', type=int, required=True)
    parser.add_argument('-sd', '--seed', type=int, required=True)
    args = parser.parse_args()

    if args.selected_fold < 0 or args.k_folds < 0 or args.selected_fold >= args.k_folds:
        raise Exception('Invalid arguments, the following must be true: k,s > 0, s < k')

    utils.log(
        'Arguments:\n' +
        'data dir={}\n'.format(args.data_dir) +
        'k-folds={}\n'.format(args.k_folds) +
        'selected fold={}\n'.format(args.selected_fold) +
        'seed={}'.format(args.seed)
    )

    dataset = Dataset(args.data_dir, args.k_folds, args.seed)

    training, _ = dataset.get_train_and_test_by_fold_id(args.selected_fold)
    utils.log('Training datasets are {}'.format(training))

    training_dirs = []
    for t in training:
        training_dirs.append(join(args.data_dir, t))

    bscan_paths = utils.get_bscan_paths(training_dirs)

    utils.log('Calculating training data mean')
    mean = utils.get_mean(bscan_paths)
    utils.log('Done calculating training data mean')

    utils.log('Calculating training data standard deviation')
    std = utils.get_standard_deviation(bscan_paths, mean)
    utils.log('Done calculating training data standard deviation')

    utils.log(
        'Final Results:\n' +
        'data dir={}\n'.format(args.data_dir) +
        'k-folds={}\n'.format(args.k_folds) +
        'selected fold={}\n'.format(args.selected_fold) +
        'seed={}\n'.format(args.seed) +
        'mean={}\n'.format(mean) +
        'std.={}'.format(std)
    )


if __name__ == '__main__':
    main()
