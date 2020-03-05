from configs.parameters import LAMBDA, EXPERIMENT
import tensorflow as tf
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
plt.gray()


# The cGAN loss function L_cGAN is maximized when the discriminator correctly
# predicts D(x,y) = 1 and D(x,G(x,z)) = 0.
# It is simply binary cross-entropy loss, negated to become a max function:
#
# L_cGAN(G, D) = E_{x,y}[log(D(x,y))] + E_{x,z}[log(1-D(x,G(x,z)))]
# where:
# x: BScan input
# y: true OMAG
# z: random noise

# The discriminator seeks to maximize L_cGAN.
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Discriminator should minimize discriminator_loss, which is a negation of
# L_cGAN.
#
# disc_real_output: D(x,y)
# disc_generated_output: D(x,G(x,z))
def discriminator_loss(disc_real_output, disc_generated_output):
    # "true values" part of the binary cross-entropy loss
    # -log(D(x,y))
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    # "false values" part of the binary cross-entropy loss
    # -log(1-D(x,G(x,z)))
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    # -log(D(x,y)) - log(1-D(x,G(x,z)))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


# The generator seeks to maximize log(D(G(x,z))), a non-saturating re-framing
# of minimize log(1-D(G(x,z))).
# (see https://arxiv.org/abs/1711.10337 for a discussion of non-saturated
# generator loss)
#
# To minimize the Euclidean distance between the real and generated images,
# we add a weighted L1 loss (pix2pix found that L2 leads to more blurring).
def generator_loss(disc_generated_output, gen_output, target):
    # -log(D(G(x,z)))
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # |y - G(x,z)|
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss


@tf.function
def train_step(model_state, input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = model_state.generator(input_image, training=True)

        disc_real_output = model_state.discriminator([input_image, target], training=True)
        disc_generated_output = model_state.discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, model_state.generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, model_state.discriminator.trainable_variables)

    model_state.generator_optimizer.apply_gradients(zip(generator_gradients, model_state.generator.trainable_variables))
    model_state.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, model_state.discriminator.trainable_variables))

    return gen_loss, disc_loss


def train_epoch(train_dataset, model_state, epoch_num):
    index = 0
    gen_loss_sum = 0
    disc_loss_sum = 0
    for input_image, target in train_dataset:
        gen_loss, disc_loss = train_step(model_state, input_image, target)
        gen_loss_sum += gen_loss
        disc_loss_sum += disc_loss
        index += 1

        # log info to Comet ML
        EXPERIMENT.log_metric("avg_gen_loss", gen_loss_sum / index, epoch=epoch_num, step=index)
        EXPERIMENT.log_metric("avg_disc_loss", disc_loss_sum / index, epoch=epoch_num, step=index)
