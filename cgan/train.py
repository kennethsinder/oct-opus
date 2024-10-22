# August 2020 - cgan/train.py
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


def train_epoch(tbd_writer, train_dataset, model_state, epoch_num, fold_num):
    for input_image, target in train_dataset:
        gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(model_state, input_image, target)

        # log info to tensorboard
        with tbd_writer.as_default():
            tf.summary.scalar('gen_total_loss_fold_{}'.format(fold_num), gen_total_loss, step=epoch_num)
            tf.summary.scalar('gen_gan_loss_fold_{}'.format(fold_num), gen_gan_loss, step=epoch_num)
            tf.summary.scalar('gen_l1_loss_fold_{}'.format(fold_num), gen_l1_loss, step=epoch_num)
            tf.summary.scalar('disc_loss_fold_{}'.format(fold_num), disc_loss, step=epoch_num)

            # While we're experimenting with the best way to leverage Tensorboard,
            # this code logs the 4 losses every 200 training steps, which is
            # more frequent than the above logging, which is just every epoch.
            if not model_state.global_index % 200:
                tf.summary.scalar('gen_total_loss_granular_fold_{}'.format(fold_num), gen_total_loss,
                                  step=model_state.global_index)
                tf.summary.scalar('gen_gan_loss_granular_fold_{}'.format(fold_num), gen_gan_loss,
                                  step=model_state.global_index)
                tf.summary.scalar('gen_l1_loss_granular_fold_{}'.format(fold_num), gen_l1_loss,
                                  step=model_state.global_index)
                tf.summary.scalar('disc_loss_granular_fold_{}'.format(fold_num), disc_loss,
                                  step=model_state.global_index)

        model_state.global_index += 1
