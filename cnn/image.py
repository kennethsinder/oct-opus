import io
import numpy as np
import tensorflow as tf
from PIL import Image

from cnn.parameters import PIXEL_DEPTH


def load(path, data_format='channels_last'):
    """ (str, str) -> tensorflow.python.framework.ops.EagerTensor
    Decodes a grayscale PNG, returns a tensor containing the image.
    Shape of tensor depends on data_format:
        - 'channels_last' returns [H,W,C]
        - 'channels_first' returns [C,H,W]
    """
    if not data_format == 'channels_last' and not data_format == 'channels_first':
        raise Exception('data_format must be either \'channels_first\' or \'channels_last\'')

    image = Image.open(path)
    output = io.BytesIO()
    image.save(output, format='png')
    img = tf.image.decode_png(output.getvalue(), channels=1)

    if data_format == 'channels_first':
        img = tf.transpose(img, [2,0,1]) # move channels first

    return img / (PIXEL_DEPTH - 1)


def save(img, path, data_format):
    """ (numpy.ndarray, str, str) -> None
    Saves the given image to the given path.
    We assume the shape of the image based on data_format:
        - 'channels_last' assumes shape of [H,W,C]
        - 'channels_first' assumes shape of [C,H,W]
    """
    if not data_format == 'channels_last' and not data_format == 'channels_first':
        raise Exception('data_format must be either \'channels_first\' or \'channels_last\'')

    # make sure values are in the interval [0, 1]
    img = np.clip(img, 0, 1)

    # format image
    if data_format == 'channels_first':
        img = tf.transpose(img, [1,2,0]) # move channels last
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
