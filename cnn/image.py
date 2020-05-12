import tensorflow as tf
import numpy as np

from cnn.parameters import IMAGE_DIM, NUM_SLICES, PIXEL_DEPTH, SLICE_WIDTH


def load(path):
    """ (str) -> tensorflow.python.framework.ops.EagerTensor
    Decodes a grayscale PNG, returns a 2D tensor.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def save(img, path):
    """ (numpy.ndarray, str) -> None
    Saves the given image to the given path.
    """
    # format image
    img *= PIXEL_DEPTH - 1

    # save image
    encoded_img = tf.image.encode_png(tf.dtypes.cast(img, tf.uint8))
    tf.io.write_file(path, encoded_img)


def resize(image, height, width):
    """ (tensorflow.python.framework.ops.EagerTensor)
            -> tensorflow.python.framework.ops.EagerTensor
    """
    return tf.image.resize(
        image,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

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


def connect(slices):
    """ (numpy.ndarray) -> numpy.ndarray
    """
    return np.concatenate(slices, axis=1)
