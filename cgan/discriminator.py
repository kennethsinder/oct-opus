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
import tensorflow_addons as tfa

from cgan.parameters import IMAGE_DIM
from cgan.sampling import downsample


def discriminator():
    """
    Discriminator expects two inputs (generated, target) with dimensions
    [batch size, IMAGE_DIM, IMAGE_DIM, 1], and is built for a batch size
    of 1. (Replace InstanceNormalization with BatchNormalization if you
    intend on extending this model to handle batched data.)

    This discriminator is a PatchGAN with a receptive field of 70x70.
    https://fomoro.com/research/article/receptive-field-calculator#4,2,1,VALID;4,2,1,VALID;4,2,1,VALID;4,1,1,VALID;4,1,1,VALID
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IMAGE_DIM, IMAGE_DIM, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMAGE_DIM, IMAGE_DIM, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 512, 512, 2)

    down1 = downsample(64, 4, apply_norm=False)(x)  # (bs, 256, 256, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 128, 128, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 64, 64, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 66, 66, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 63, 63, 512)

    norm = tfa.layers.InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 65, 65, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 62, 62, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# Discriminator should minimize discriminator_loss, which is a negation of
# L_cGAN.
#
# disc_real_output: D(x,y)
# disc_generated_output: D(x,G(x,z))
def discriminator_loss(loss_object, disc_real_output, disc_generated_output):
    # "true values" part of the binary cross-entropy loss
    # -log(D(x,y))

    # 0.9 label is to implement one-sided label smoothing, see
    # Goodfellow https://arxiv.org/pdf/1701.00160.pdf (citing Salimans et. al)
    real_loss = loss_object(tf.ones_like(disc_real_output) * 0.9, disc_real_output)

    # "false values" part of the binary cross-entropy loss
    # -log(1-D(x,G(x,z)))
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    # -log(D(x,y)) - log(1-D(x,G(x,z)))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss
