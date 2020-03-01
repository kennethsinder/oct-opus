from configs.parameters import LAMBDA, EXPERIMENT
import tensorflow as tf
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
plt.gray()


# TODO: remove global variables
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
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
    gen_loss_sum = 0
    disc_loss_sum = 0
    for input_image, target in train_dataset:
        gen_loss, disc_loss = train_step(model_state, input_image, target)
        gen_loss_sum += gen_loss
        disc_loss_sum += disc_loss
        model_state.current_training_step += 1

        # log info to Comet ML
        index = model_state.current_training_step
        EXPERIMENT.log_metric("avg_gen_loss", gen_loss_sum / index, epoch=epoch_num, step=index)
        EXPERIMENT.log_metric("avg_disc_loss", disc_loss_sum / index, epoch=epoch_num, step=index)
