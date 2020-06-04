import glob
from os import makedirs
from os.path import basename, isfile, join, splitext

import tensorflow as tf

import cnn.image as image
from cnn.parameters import (
    BUFFER_SIZE,
    IMAGE_DIM,
    NUM_SLICES,
    SLICE_WIDTH,
)


def get_bscan_paths(data_dirs):
    """ (list) -> list
    """
    bscan_paths = []

    for data_dir in data_dirs:
        bscan_paths.extend(glob.glob(join(data_dir,  'xzIntensity', '*.png')))

    if not bscan_paths:
        raise Exception('No B-scan images were found.')
    return bscan_paths


def load_dataset(bscan_paths, batch_size, shuffle=True):
    """ (list, int, bool, bool)
            -> tensorflow.python.data.ops.dataset_ops.BatchDataset, int
    Returns a generator dataset & the number of batches. Number of batches does
    not include batches with size less than batch_size.
    """

    output_shape = tf.TensorShape((NUM_SLICES, IMAGE_DIM, SLICE_WIDTH, 1))
    dataset = tf.data.Dataset.from_generator(
        lambda: map(get_slices, bscan_paths),
        output_types=(tf.float32, tf.float32),
        output_shapes=(output_shape, output_shape)
    )

    # silently drop data that causes errors (e.g. corresponding OMAG file doesn't exist)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    # need to unbatch so that each image slice is its own input, instead of
    # having 4 slices grouped together as one input
    dataset = dataset.unbatch()

    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE)

    # re-batch the images into the appropriate batch size
    dataset = dataset.batch(batch_size)

    # it's possible the last batch has a size less than batch_size, then it will
    # need to be removed
    num_batches = (len(bscan_paths) * NUM_SLICES) // batch_size
    dataset = dataset.take(num_batches)

    return dataset, num_batches


def get_slices(bscan_path):
    """ (str) -> tensorflow.python.framework.ops.EagerTensor,
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

    # slice images into vertical strips
    bscan_img_slices = image.slice(bscan_img)
    omag_img_slices = image.slice(omag_img)

    return bscan_img_slices, omag_img_slices


def get_num_acquisitions(data_dir):
    """ (str) -> int
    Auto-detect the number of acquisitions used for the data set in the
    folder identified by `data_dir`. Usually this will return
    the integer 1 or 4 (4 acquisitions is normal for OMAG).
    """
    bscan_paths = glob.glob(join(data_dir, 'xzIntensity', '*'))
    omag_paths = glob.glob(join(data_dir, 'OMAG Bscans', '*'))
    return int(round(len(bscan_paths) / float(len(omag_paths))))


def bscan_num_to_omag_num(bscan_num, num_acquisitions):
    """ (int, int) -> int
    """
    return ((bscan_num - 1) // num_acquisitions) + 1