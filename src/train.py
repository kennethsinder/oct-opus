import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.gray()
import tensorflow as tf

from configs.parameters import LAMBDA, IMAGE_LOG_INTERVAL, START_ROW, END_ROW

# TODO: remove global variables
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def slice_tensor(t):
    dim_1, dim_2, dim_3, dim_4 = t.get_shape().as_list()
    start_row = START_ROW if START_ROW is not None else 0
    end_row = END_ROW if END_ROW is not None else (dim_2 // 2)
    return tf.slice(t, (0, start_row, 0, 0), (dim_1, end_row - start_row, dim_3, dim_4))


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target, should_slice=False):
    gen_output_c = slice_tensor(gen_output) if should_slice else gen_output
    target_c = slice_tensor(target) if should_slice else target

    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target_c - gen_output_c))
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

    return gen_output, disc_real_output, disc_generated_output, gen_loss, disc_loss


def generate_images(model, test_input, tar, epoch_num):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(tf.squeeze(display_list[i]) * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('comparison_epoch_{}.png'.format(epoch_num))


def train_epoch(train_dataset, model_state, writer, epoch_num):
    gen_loss_sum = 0
    disc_loss_sum = 0
    idx = 0
    for input_image, target in train_dataset:
        gen_output, disc_real_output, disc_generated_output, gen_loss, disc_loss = train_step(model_state, input_image,
                                                                                              target)
        if idx % IMAGE_LOG_INTERVAL == 0:
            print('\tStep {}: logging images to Tensorboard...'.format(idx))
            with writer.as_default():
                tf.summary.image('{}_input_image'.format(epoch_num), input_image, step=idx)
                tf.summary.image('{}_target'.format(epoch_num), target, step=idx)
                tf.summary.image('{}_gen_output'.format(epoch_num), gen_output, step=idx)
                tf.summary.image('{}_disc_real_output'.format(epoch_num), disc_real_output, step=idx)
                tf.summary.image('{}_disc_generated_output'.format(epoch_num), disc_generated_output, step=idx)
        gen_loss_sum += gen_loss
        disc_loss_sum += disc_loss
        idx += 1
    with writer.as_default():
        tf.summary.scalar('avg_gen_loss', gen_loss_sum / idx, step=epoch_num)
        tf.summary.scalar('avg_disc_loss', disc_loss_sum / idx, step=epoch_num)


def train(model_state, writer, epoch_num):
    print('Starting epoch {} ...'.format(epoch_num))
    start = time.time()

    train_epoch(model_state.train_dataset, model_state, writer, epoch_num)
    for inp, tar in model_state.test_dataset.take(1):
        generate_images(model_state.generator, inp, tar, epoch_num)
    model_state.save_checkpoint()

    print('Time taken for epoch {} is {} sec\n'.format(epoch_num, time.time() - start))
