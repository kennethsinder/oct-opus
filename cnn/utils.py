import glob
import math
import random
from datetime import datetime
from itertools import repeat
from os.path import basename, join, splitext

import tensorflow as tf

import cnn.image as image
from cnn.parameters import IMAGE_DIM, NUM_DATASETS, NUM_IMAGES_PER_DATASET

from cgan.random import resize, random_crop, random_noise, random_jitter


def get_mean(bscan_paths):
    mean = 0
    count = 0
    for path in bscan_paths:
        img = image.load(path)
        for row in tf.squeeze(img).numpy():
            for val in row:
                mean += val
                count += 1
    mean /= count
    return mean


def get_standard_deviation(bscan_paths, mean):
    std = 0
    count = 0
    for path in bscan_paths:
        img = image.load(path)
        for row in tf.squeeze(img).numpy():
            for val in row:
                std += math.pow(val - mean, 2)
                count += 1
    std /= count
    return math.sqrt(std)


def separate_training_testing(root_data_dir, split, seed):
    """ (str, float, int) -> list, list
    """
    data_names = []
    for data_dir in glob.glob(join(root_data_dir, '*')):
        data_names.append(basename(data_dir))
    data_names.sort()

    random.seed(seed)

    training_data_names = random.sample(data_names, int(split * len(data_names)))
    testing_data_names = [d for d in data_names if d not in training_data_names]

    training_dirs = []
    for name in training_data_names:
        training_dirs.append(join(root_data_dir, name))

    testing_dirs = []
    for name in testing_data_names:
        testing_dirs.append(join(root_data_dir, name))

    return training_dirs, testing_dirs


def get_bscan_paths(data_dirs):
    """ (list) -> list
    """
    bscan_paths = []

    for dir in data_dirs:
        bscan_paths.extend(glob.glob(join(dir, 'xzIntensity', '[0-9]*.png')))

    if not bscan_paths:
        raise Exception('No B-scan images were found.')
    return bscan_paths


def load_dataset(bscan_paths, batch_size, num_slices, mean, standard_deviation, shuffle):
    """ (list, int, float, float, bool)
            -> tensorflow.python.data.ops.dataset_ops.BatchDataset, int
    Returns a generator dataset & the number of batches. Images are of the form
    [C,H,W]. Number of batches does not include batches with size less than batch_size.
    """

    output_shape = tf.TensorShape((num_slices, 1, IMAGE_DIM, IMAGE_DIM // num_slices))
    dataset = tf.data.Dataset.from_generator(
        lambda: map(
            get_slices,
            bscan_paths,
            repeat(num_slices),
            repeat(mean),
            repeat(standard_deviation)
        ),
        output_types=(tf.float32, tf.float32),
        output_shapes=(output_shape, output_shape)
    )

    # silently drop data that causes errors (e.g. corresponding OMAG file doesn't exist)
    # dataset = dataset.apply(tf.data.experimental.ignore_errors())

    # need to unbatch so that each image slice is its own input, instead of
    # having 4 slices grouped together as one input
    dataset = dataset.unbatch()

    if shuffle:
        dataset = dataset.shuffle(NUM_DATASETS * NUM_IMAGES_PER_DATASET * num_slices)

    # re-batch the images into the appropriate batch size
    # set drop_remainder to True to exclude any batches that are
    # less than batch_size
    dataset = dataset.batch(batch_size, drop_remainder=True)

    num_batches = (len(bscan_paths) * num_slices) // batch_size

    return dataset, num_batches


def get_slices(bscan_path, num_slices, mean, standard_deviation):
    """ (str, float, float) -> tensorflow.python.framework.ops.EagerTensor,
                 tensorflow.python.framework.ops.EagerTensor
    Returns a pair of tensors containing the given B-scan slices and their
    corresponding OMAG slices.
    |bscan_path| should be in directory 'xzIntensity', and its parent directory
    hould contain 'OMAG Bscans'.
    """

    bscan_img = image.load(bscan_path)

    dir_path, bscan_num = splitext(bscan_path)[0].split('xzIntensity/')
    bscan_num = int(bscan_num)

    omag_num = bscan_num_to_omag_num(bscan_num, get_num_acquisitions(dir_path))
    omag_img = image.load(join(dir_path, 'OMAG Bscans', '{}.png'.format(omag_num)))

    # resize images
    bscan_img = image.resize(bscan_img, IMAGE_DIM, IMAGE_DIM)
    omag_img = image.resize(omag_img, IMAGE_DIM, IMAGE_DIM)

    # move to channels first
    bscan_img = tf.transpose(bscan_img, [2,0,1])
    omag_img = tf.transpose(omag_img, [2,0,1])

    # centre the input image only
    bscan_img = tf.math.subtract(bscan_img, mean)
    bscan_img = tf.math.divide(bscan_img, standard_deviation)

    # slice images into vertical strips
    bscan_img_slices = tf.convert_to_tensor(tf.split(bscan_img, num_slices, 2))
    omag_img_slices = tf.convert_to_tensor(tf.split(omag_img, num_slices, 2))

    return bscan_img_slices, omag_img_slices


def load_augmented_dataset(bscan_paths, batch_size, num_slices, use_random_jitter, use_random_noise, shuffle):
    """ (list, int, float, bool, bool, bool)
            -> tensorflow.python.data.ops.dataset_ops.BatchDataset, int
    Returns a generator dataset & the number of batches. Images are of the form
    [C,H,W]. Number of batches does not include batches with size less than batch_size.
    """

    output_shape = tf.TensorShape((num_slices, 1, IMAGE_DIM, IMAGE_DIM // num_slices))
    dataset = tf.data.Dataset.from_generator(
        lambda: map(
            get_augmented_slices,
            bscan_paths,
            repeat(num_slices),
            repeat(use_random_jitter),
            repeat(use_random_noise)
        ),
        output_types=(tf.float32, tf.float32),
        output_shapes=(output_shape, output_shape)
    )

    # silently drop data that causes errors (e.g. corresponding OMAG file doesn't exist)
    # dataset = dataset.apply(tf.data.experimental.ignore_errors())

    # need to unbatch so that each image slice is its own input, instead of
    # having 4 slices grouped together as one input
    dataset = dataset.unbatch()

    if shuffle:
        dataset = dataset.shuffle(NUM_DATASETS * NUM_IMAGES_PER_DATASET * num_slices)

    # re-batch the images into the appropriate batch size
    # set drop_remainder to True to exclude any batches that are
    # less than batch_size
    dataset = dataset.batch(batch_size, drop_remainder=True)

    num_batches = (len(bscan_paths) * num_slices) // batch_size

    return dataset, num_batches



def get_augmented_slices(bscan_path, num_slices, use_random_jitter, use_random_noise):
    """ (str, float, bool, bool) -> tensorflow.python.framework.ops.EagerTensor,
                 tensorflow.python.framework.ops.EagerTensor
    Returns a pair of tensors containing the given B-scan slices and their
    corresponding OMAG slices.
    |bscan_path| should be in directory 'xzIntensity', and its parent directory
    hould contain 'OMAG Bscans'.
    """

    # random jitter angle
    angle = 0
    if use_random_jitter and tf.random.uniform(()) > 0.8:
        angle = random.randint(0, 45)

    bscan_img = image.load(bscan_path, angle=angle, contrast_factor=1.85)

    dir_path, bscan_num = splitext(bscan_path)[0].split('xzIntensity/')
    bscan_num = int(bscan_num)

    omag_num = bscan_num_to_omag_num(bscan_num, get_num_acquisitions(dir_path))
    omag_img = image.load(join(dir_path, 'OMAG Bscans', '{}.png'.format(omag_num)), angle=angle, contrast_factor=1.85)

    # random jitter
    if use_random_jitter:
        bscan_img, omag_img = random_jitter(bscan_img, omag_img)
    else:
        bscan_img, omag_img = resize(bscan_img, omag_img, IMAGE_DIM, IMAGE_DIM)

    # random noise
    if use_random_noise:
        # don't add noise to the omag image
        bscan_img = random_noise(bscan_img)

    # move to channels first
    bscan_img = tf.transpose(bscan_img, [2,0,1])
    omag_img = tf.transpose(omag_img, [2,0,1])

    # slice images into vertical strips
    bscan_img_slices = tf.convert_to_tensor(tf.split(bscan_img, num_slices, 2))
    omag_img_slices = tf.convert_to_tensor(tf.split(omag_img, num_slices, 2))

    return bscan_img_slices, omag_img_slices


def get_num_acquisitions(data_dir):
    """ (str) -> int
    Auto-detect the number of acquisitions used for the data set in the
    folder identified by `data_dir`. Usually this will return
    the integer 1 or 4 (4 acquisitions is normal for OMAG).
    """
    bscan_paths = glob.glob(join(data_dir, 'xzIntensity', '[0-9]*.png'))
    omag_paths = glob.glob(join(data_dir, 'OMAG Bscans', '[0-9]*.png'))
    return int(round(len(bscan_paths) / float(len(omag_paths))))


def bscan_num_to_omag_num(bscan_num, num_acquisitions):
    """ (int, int) -> int
    """
    return ((bscan_num - 1) // num_acquisitions) + 1


def log(message):
    """ (str) -> None
    Prints a message to standard output, add the current time to the message.
    """
    print('[{}] {}'.format(datetime.now().strftime('%b %d, %Y - %H:%M:%S'), message))
