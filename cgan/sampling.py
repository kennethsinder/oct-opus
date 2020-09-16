# August 2020 - cgan/sampling.py
#
# This code is an adaptation of the approach from the 2017 pix2pix paper with some of our own additional small
# tweaks to the generator and discriminator model to suit our application.
#
#     Isola, P., Zhu, J., Zhou, T., Efros, A. A.,
#     "Image-to-Image Translation with Conditional Adversarial Networks,"
#     IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, 5967-5976 (2017).
#
# Significant additional credit also goes to TensorFlow's pix2pix tutorial page:
#
#     https://www.tensorflow.org/tutorials/generative/pix2pix
#
# We adapted a lot of code snippets from that tutorial, making changes as necessary to support our folder structure
# and use cases, as well as small model changes and refactoring the code over multiple files for maintainability.

import tensorflow as tf
import tensorflow_addons as tfa


def downsample(filters, size, apply_norm=True, padding_mode='REFLECT'):
    """
    From Conv2D documentation
    filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Lambda(
        lambda input: tf.pad(input, [[0, 0], [2, 1], [2, 1], [0, 0]],
                             mode=padding_mode)
    ))
    result.add(tf.keras.layers.Conv2D(filters, size,
                                      strides=2,
                                      padding='valid',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    if apply_norm:
        result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False, padding_mode='REFLECT'):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    # https://distill.pub/2016/deconv-checkerboard/
    # https://www.machinecurve.com/index.php/2019/12/11/upsampling2d-how-to-use-upsampling-with-keras/
    result.add(tf.keras.layers.UpSampling2D(2, interpolation='nearest'))
    result.add(tf.keras.layers.Lambda(
        lambda input: tf.pad(input, [[0, 0], [2, 1], [2, 1], [0, 0]],
                             mode=padding_mode)
    ))
    result.add(tf.keras.layers.Conv2D(filters, size, strides=1,
                                      padding='valid',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result
