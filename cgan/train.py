import tensorflow as tf

from cgan.discriminator import discriminator_loss
from cgan.generator import generator_loss


@tf.function
def train_step(model_state, input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = model_state.generator(input_image, training=True)

        disc_real_output = model_state.discriminator([input_image, target], training=True)
        disc_generated_output = model_state.discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(model_state.loss_object, disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(model_state.loss_object, disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, model_state.generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, model_state.discriminator.trainable_variables)

    model_state.generator_optimizer.apply_gradients(zip(generator_gradients, model_state.generator.trainable_variables))
    model_state.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, model_state.discriminator.trainable_variables))

    return gen_loss, disc_loss


def train_epoch(tbd_writer, train_dataset, model_state, epoch_num):
    gen_loss_sum = 0
    disc_loss_sum = 0
    index = 0
    for input_image, target in train_dataset:
        gen_loss, disc_loss = train_step(model_state, input_image, target)
        gen_loss_sum += gen_loss
        disc_loss_sum += disc_loss
        model_state.current_training_step += 1
        index += 1

        # log info to tensorboard
        with tbd_writer.as_default():
            tf.summary.scalar('avg_gen_loss', gen_loss_sum / index, step=epoch_num)
            tf.summary.scalar('avg_disc_loss', disc_loss_sum / index, step=epoch_num)
