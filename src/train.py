import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

# TODO: remove global variables
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(
        disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(
        disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss


@tf.function
def train_step(model_state, input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = model_state.generator(input_image, training=True)

        disc_real_output = model_state.discriminator(
            [input_image, target], training=True)
        disc_generated_output = model_state.discriminator(
            [input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        gen_loss, model_state.generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(
        disc_loss, model_state.discriminator.trainable_variables)

    model_state.generator_optimizer.apply_gradients(
        zip(generator_gradients, model_state.generator.trainable_variables))
    model_state.discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, model_state.discriminator.trainable_variables))


def generate_images(model, test_input, tar):
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
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(tf.squeeze(display_list[i]) * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def train_epoch(train_dataset, model_state):
    for input_image, target in train_dataset:
        train_step(model_state, input_image, target)


def train(model_state):
    print('Starting epoch')
    start = time.time()

    train_epoch(model_state.train_dataset, model_state)
    model_state.save_checkpoint()

    print('Time taken for epoch is {} sec\n'.format(
        time.time() - start))
