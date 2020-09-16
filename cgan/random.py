# August 2020 - cgan/random.py
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

from cgan.parameters import IMAGE_DIM


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.concat([input_image, real_image], axis=2)
    cropped_image = tf.image.random_crop(
        stacked_image,
        size=[IMAGE_DIM, IMAGE_DIM, input_image.shape[-1]+real_image.shape[-1]]
    )
    return (cropped_image[..., :input_image.shape[-1]],
            cropped_image[..., input_image.shape[-1]:])


def random_rot(input_image, real_image):
    stacked_image = tf.concat([input_image, real_image], axis=2)
    rot_image = tf.keras.preprocessing.image.random_rotation(
        stacked_image.numpy(), 45,
        row_axis=0,
        col_axis=1,
        channel_axis=2,
        fill_mode='reflect'
    )
    return (tf.convert_to_tensor(rot_image[..., :input_image.shape[-1]]),
            tf.convert_to_tensor(rot_image[..., input_image.shape[-1]:]))


def random_noise(input_image):
    if tf.random.uniform(()) > 0.5:
        converted_image = tf.image.convert_image_dtype(input_image, tf.float32)
        noise = tf.random.normal(input_image.shape, stddev=0.1)
        input_image = tf.image.convert_image_dtype(converted_image + noise, tf.uint8)
    return input_image


def random_jitter(input_image, real_image):
    if tf.random.uniform(()) > 0.5:
        input_image, real_image = resize(input_image, real_image, IMAGE_DIM + 30, IMAGE_DIM + 30)
        input_image, real_image = random_crop(input_image, real_image)
    else:
        input_image, real_image = resize(input_image, real_image, IMAGE_DIM, IMAGE_DIM)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    if tf.random.uniform(()) > 0.8:
        input_image, real_image = random_rot(input_image, real_image)

    return input_image, real_image
