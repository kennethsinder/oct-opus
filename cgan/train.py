import tensorflow as tf

from cgan.discriminator import discriminator_loss
from cgan.generator import generator_loss


@tf.function
def train_step(model_state, input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = model_state.generator(input_image, training=True)

        disc_real_output = model_state.discriminator([input_image, target], training=True)
        disc_generated_output = model_state.discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(model_state.loss_object, disc_generated_output,
                                                                   gen_output, target)
        disc_loss = discriminator_loss(model_state.loss_object, disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, model_state.generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, model_state.discriminator.trainable_variables)

    model_state.generator_optimizer.apply_gradients(zip(generator_gradients, model_state.generator.trainable_variables))
    model_state.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                            model_state.discriminator.trainable_variables))

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def train_epoch(tbd_writer, train_dataset, model_state, epoch_num):
    for input_image, target in train_dataset:
        gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(model_state, input_image, target)

        # log info to tensorboard
        with tbd_writer.as_default():
            tf.summary.scalar('avg_gen_total_loss', gen_total_loss, step=epoch_num)
            tf.summary.scalar('avg_gen_gan_loss', gen_gan_loss, step=epoch_num)
            tf.summary.scalar('avg_l1_loss', gen_l1_loss, step=epoch_num)
            tf.summary.scalar('avg_disc_loss', disc_loss, step=epoch_num)

            # While we're experimenting with the best way to leverage Tensorboard,
            # this code logs the 4 losses every 200 training steps, which is
            # more frequent than the above logging, which is just every epoch.
            if not model_state.global_index % 200:
                tf.summary.scalar('avg_gen_total_loss_granular', gen_total_loss, step=model_state.global_index)
                tf.summary.scalar('avg_gen_gan_loss_granular', gen_gan_loss, step=model_state.global_index)
                tf.summary.scalar('avg_l1_loss_granular', gen_l1_loss, step=model_state.global_index)
                tf.summary.scalar('avg_disc_loss_granular', disc_loss, step=model_state.global_index)

        model_state.global_index += 1
