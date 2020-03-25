import tensorflow as tf

from cgan.parameters import IMAGE_DIM, LAMBDA, LAYER_BATCH
from cgan.sampling import downsample, upsample

# Generator expect inputs with dimensions
# [batch size, IMAGE_DIM, IMAGE_DIM, LAYER_BATCH]
def generator():
    inputs = tf.keras.layers.Input(shape=[IMAGE_DIM, IMAGE_DIM, LAYER_BATCH])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(LAYER_BATCH, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')

    concat = tf.keras.layers.Concatenate()

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

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
