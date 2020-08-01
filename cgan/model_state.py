import io
import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf

from typing import Set

from cgan.discriminator import discriminator, discriminator_loss
from cgan.generator import generator, generator_loss
from cgan.parameters import LAYER_BATCH
from cgan.utils import generate_inferred_enface, get_dataset


class ModelState:
    def __init__(self, name: str,
                 exp_dir: str,
                 data_dir: str,
                 holdout_set: Set[str]):
        self.name = name

        # optimizers
        self.discriminator_optimizer = tf.keras.optimizers.Adam(5e-4,
                                                                beta_1=0.5)
        self.generator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)

        # generator and discriminator
        self.generator = generator()
        self.discriminator = discriminator()

        # paths
        self.output_dir = os.path.join(exp_dir, f"{self.name}_predictions")
        os.makedirs(self.output_dir, exist_ok=False)

        # dataset
        self.holdout_set = holdout_set
        self.holdout_data = get_dataset(data_dir, holdout_set)

        # current step, across all epochs
        self.global_step = tf.Variable(1, name="step", dtype=tf.int64)

        # current epoch
        self.epoch = tf.Variable(1, name="epoch", dtype=tf.int64)

        # TensorBoard logger
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(exp_dir, "logs", self.name))

        # checkpoints
        self.checkpoint_dir = os.path.join(exp_dir, f"{self.name}_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=False)
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
            global_step=self.global_step,
            epoch=self.epoch
        )

        # The cGAN loss function L_cGAN is maximized when the discriminator
        # correctly predicts D(x,y) = 1 and D(x,G(x,z)) = 0.
        # It is simply binary cross-entropy loss, negated to become a max
        # function:
        #
        # L_cGAN(G, D) = E_{x,y}[log(D(x,y))] + E_{x,z}[log(1-D(x,G(x,z)))]
        # where:
        # x: BScan input
        # y: true OMAG
        # z: random noise

        # The discriminator seeks to maximize L_cGAN.
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def end_epoch_and_checkpoint(self):
        self.checkpoint.save(file_prefix=os.path.join(self.checkpoint_dir,
                                                      f"epoch-{self.epoch.numpy()}"))
        self.epoch.assign_add(1)

    def restore_from_checkpoint(self, exp_dir, predict_only=False):
        restore_dir = os.path.join(exp_dir,
                                   os.path.basename(self.checkpoint_dir))
        latest = tf.train.latest_checkpoint(restore_dir)

        if latest is None:
            return

        if predict_only:
            # Suppress warnings that training-only parts of the checkpoint
            # are never used.
            self.checkpoint.restore(latest).expect_partial()
        else:
            self.checkpoint.restore(latest)

        self.epoch.assign_add(1)  # start next epoch

    @tf.function
    def _train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            central_input_image = input_image[:, :, :, LAYER_BATCH // 2,
                                              tf.newaxis]
            central_gen_output = gen_output[:, :, :, LAYER_BATCH // 2,
                                            tf.newaxis]

            disc_real_output = self.discriminator(
                [central_input_image, target], training=True)
            disc_generated_output = self.discriminator(
                [central_input_image, central_gen_output], training=True)

            gen_total_loss, gen_adversarial_loss, gen_l1_loss = generator_loss(
                self.loss_object, disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(self.loss_object, disc_real_output,
                                            disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients,
                self.discriminator.trainable_variables))

        return gen_total_loss, gen_adversarial_loss, gen_l1_loss, disc_loss

    def _log_loss_values(self, gen_total_loss: float,
                          gen_adversarial_loss: float, gen_l1_loss: float,
                          disc_loss: float) -> None:
        with self.summary_writer.as_default():
            tf.summary.scalar("gen_total_loss",
                              gen_total_loss,
                              step=self.global_step)
            tf.summary.scalar("gen_adversarial_loss",
                              gen_adversarial_loss,
                              step=self.global_step)
            tf.summary.scalar("gen_l1_loss",
                              gen_l1_loss,
                              step=self.global_step)
            tf.summary.scalar("disc_loss", disc_loss, step=self.global_step)

    def _log_output_comparison(self) -> None:
        _, images = list(self.holdout_data.take(1))[0]
        inp = images[0]
        tar = images[1]

        # the `training=True` is intentional here since
        # we want the batch statistics while running the model
        # on the test dataset. If we use training=False, we will get
        # the accumulated statistics learned from the training dataset
        # (which we don't want)
        pred = self.generator(inp, training=True)
        pred = pred[0, :, :, LAYER_BATCH // 2, tf.newaxis]

        inp = inp[0, :, :, LAYER_BATCH // 2, tf.newaxis]
        tar = tar[0, ...]

        plt.figure(figsize=(15, 15))

        display_list = [inp, tar, pred]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i, img in enumerate(display_list):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(tf.squeeze(img) * 0.5 + 0.5, cmap='gray')
            plt.axis('off')

        with self.summary_writer.as_default():
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt_img = tf.image.decode_png(buf.getvalue(), channels=4)
            tf.summary.image("output_comparison",
                              plt_img[tf.newaxis, ...],
                              step=self.global_step)
            plt.close()

    def train_step(self, input_image, target_image) -> None:
        loss_values = self._train_step(input_image, target_image)

        if self.global_step % 200 == 0:
            self._log_loss_values(*loss_values)

        if self.global_step % 1000 == 0:
            self._log_output_comparison()

        self.global_step.assign_add(1)
