import glob
from os.path import basename, join, normpath, splitext

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from cnn.parameters import (
    BUFFER_SIZE,
    IMAGE_DIM,
    NUM_SLICES,
    SLICE_WIDTH
)


def get_bscan_paths(dataset_dirs):
    """ (list) -> list
    """
    bscan_paths = []

    for dataset_dir in dataset_dirs:
        bscan_paths.extend(glob.glob(join(dataset_dir,  'xzIntensity', '*.png')))
        # for path in glob.glob(join(dataset_dir,  'xzIntensity', '*.png')):
        #     bscan_paths.append(path)

    if not bscan_paths:
        raise Exception('No B-scan images were found.')
    return bscan_paths


def load_dataset(bscan_paths, batch_size, repeat=True, shuffle=True):
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


def shuffle(dataset, batch_size):
    """(tensorflow.python.data.ops.dataset_ops.BatchDataset)
            -> tensorflow.python.data.ops.dataset_ops.BatchDataset
    """
    return dataset.unbatch().shuffle(BUFFER_SIZE).batch(batch_size)


def get_dataset_name(bscan_path):
    """ (str) -> str
    """
    return basename(normpath(join(bscan_path, '..', '..')))


def resize(image, height, width):
    """ (tensorflow.python.framework.ops.EagerTensor)
            -> tensorflow.python.framework.ops.EagerTensor
    """
    return tf.image.resize(
        image,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )


# Decodes a grayscale PNG, returns a 2D tensor.
def load_image(path):
    """ (str) -> tensorflow.python.framework.ops.EagerTensor
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def get_num_acquisitions(data_folder_path):
    """ (str) -> int
    Auto-detect the number of acquisitions used for the data set in the
    folder identified by `data_folder_path`. Usually this will return
    the integer 1 or 4 (4 acquisitions is normal for OMAG).
    """
    bscan_paths = glob.glob(join(data_folder_path, 'xzIntensity', '*'))
    omag_paths = glob.glob(join(data_folder_path, 'OMAG Bscans', '*'))
    return int(round(len(bscan_paths) / float(len(omag_paths))))


def bscan_num_to_omag_num(bscan_num, num_acquisitions):
    """ (int, int) -> int
    """
    return ((bscan_num - 1) // num_acquisitions) + 1


def slice(img):
    """ (tensorflow.python.framework.ops.EagerTensor)
            -> tensorflow.python.framework.ops.EagerTensor
    Returns image sliced into NUM_SLICES number of vertical slices.
    For an input tensor with shape [x, y, z], the returning tensor has shape
    [NUM_SLICES, x, y, z]
    """
    return tf.convert_to_tensor(
        tf.split(img, [SLICE_WIDTH] * NUM_SLICES, 1)
    )


def connect_slices(slices):
    """ (numpy.ndarray) -> numpy.ndarray
    """
    return np.concatenate(slices, axis=1)


def get_slices(bscan_path):
    """ (str) -> tensorflow.python.framework.ops.EagerTensor,
                 tensorflow.python.framework.ops.EagerTensor
    Returns a pair of tensors containing the given B-scan slices and their
    corresponding OMAG slices.
    |bscan_path| should be in directory 'xzIntensity', and its parent directory
    hould contain 'OMAG Bscans'.
    Scan files should be named <num>.png (no leading 0s), with a 4-to-1 ratio of
    B-scans to OMAGs.
    (OMAG Bscans/1.png corresponds to xzIntensity/{1,2,3,4}.png.)
    """

    bscan_img = load_image(bscan_path)

    dir_path, bscan_num = splitext(bscan_path)[0].split('xzIntensity/')
    bscan_num = int(bscan_num)

    omag_num = bscan_num_to_omag_num(bscan_num, get_num_acquisitions(dir_path))

    omag_img = load_image(join(dir_path, 'OMAG Bscans', '{}.png'.format(omag_num)))

    # resize images to 512x512
    bscan_img = resize(bscan_img, IMAGE_DIM, IMAGE_DIM)
    omag_img = resize(omag_img, IMAGE_DIM, IMAGE_DIM)

    # slice images into vertical strips
    bscan_img_slices = slice(bscan_img)
    omag_img_slices = slice(omag_img)

    return bscan_img_slices, omag_img_slices
