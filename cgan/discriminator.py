# August 2020 - cgan/discriminator.py
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
from cgan.sampling import downsample


def discriminator():
    """
    Discriminator and Generator expect inputs with dimensions
    [batch size, IMAGE_DIM, IMAGE_DIM, LAYER_BATCH]
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# Discriminator should minimize discriminator_loss, which is a negation of
# L_cGAN.
#
# disc_real_output: D(x,y)
# disc_generated_output: D(x,G(x,z))
def discriminator_loss(loss_object, disc_real_output, disc_generated_output):
    # "true values" part of the binary cross-entropy loss
    # -log(D(x,y))
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    # "false values" part of the binary cross-entropy loss
    # -log(1-D(x,G(x,z)))
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    # -log(D(x,y)) - log(1-D(x,G(x,z)))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss
