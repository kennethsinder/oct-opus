# August 2020 - cgan/generator.py
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

from cgan.parameters import IMAGE_DIM, LAMBDA, LAYER_BATCH
from cgan.sampling import downsample, upsample


def generator():
    """
    Generator expects a single input with dimensions
    [batch size, IMAGE_DIM, IMAGE_DIM, LAYER_BATCH], and is built for a
    batch size of 1. (Replace InstanceNormalization with
    BatchNormalization if you intend on extending this model to handle
    batched data.)
    """

    inputs = tf.keras.layers.Input(shape=[IMAGE_DIM, IMAGE_DIM, LAYER_BATCH])

    down_stack = [
        downsample(64, 4, apply_norm=False),  # (bs, 256, 256, 64)
        downsample(128, 4),  # (bs, 128, 128, 128)
        downsample(256, 4),  # (bs, 64, 64, 256)
        downsample(512, 4),  # (bs, 32, 32, 512)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4, padding_mode='SYMMETRIC'),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True,
                 padding_mode='SYMMETRIC'),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(512, 4),  # (bs, 32, 32, 1024)
        upsample(256, 4),  # (bs, 64, 64, 512)
        upsample(128, 4),  # (bs, 128, 128, 256)
        upsample(64, 4),   # (bs, 256, 256, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.Sequential()
    last.add(tf.keras.layers.UpSampling2D(2, interpolation='nearest'))
    last.add(tf.keras.layers.Lambda(
        lambda input: tf.pad(input, [[0, 0], [2, 1], [2, 1], [0, 0]],
                             mode='REFLECT')
    ))
    last.add(tf.keras.layers.Conv2D(
        LAYER_BATCH, 4,
        strides=1,
        padding='valid',
        kernel_initializer=initializer,
        activation='tanh')
    )  # (bs, 512, 512, LAYER_BATCH)

    x = inputs

    # Downsampling
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# The generator seeks to maximize log(D(G(x,z))), a non-saturating re-framing
# of minimize log(1-D(G(x,z))).
# (see https://arxiv.org/abs/1711.10337 for a discussion of non-saturated
# generator loss)
#
# To minimize the Euclidean distance between the real and generated images,
# we add a weighted L1 loss (pix2pix found that L2 leads to more blurring).
def generator_loss(loss_object, disc_generated_output, gen_output, target):
    # -log(D(G(x,z)))
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # |y - G(x,z)|
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss
