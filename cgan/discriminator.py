import tensorflow as tf
from cgan.sampling import downsample

from cgan.parameters import IMAGE_DIM

def discriminator():
    """
    Discriminator and Generator expect inputs with dimensions
    [batch size, IMAGE_DIM, IMAGE_DIM, 1]
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IMAGE_DIM, IMAGE_DIM, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMAGE_DIM, IMAGE_DIM, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

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
