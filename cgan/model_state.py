import os
from os.path import join

import tensorflow as tf

from cgan.discriminator import discriminator
from cgan.generator import generator
from cgan.utils import get_dataset
from datasets.train_and_test import train_and_test_sets


class ModelState:

    def __init__(self, EXP_DIR, all_data_path, checkpoint_dir):
        # optimizers
        self.discriminator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)
        self.generator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)

        # generator and discriminator
        self.generator = generator()
        self.discriminator = discriminator()

        # paths
        self.all_data_path = all_data_path
        self.checkpoint_dir = checkpoint_dir

        # datasets
        self.train_dataset = None
        self.test_dataset = None

        # checkpoints
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

        # Save scrambled weights before we do any training temporarily
        # so that we can reload them with model_state.reset_weights()
        # below between folds for k-folds cross-validation.
        self.generator_weights_file = join(EXP_DIR, 'generator_weights.h5')
        self.discriminator_weights_file = join(EXP_DIR, 'discriminator_weights.h5')
        self.generator.save_weights(self.generator_weights_file)
        self.discriminator.save_weights(self.discriminator_weights_file)
        self.current_training_step = 0

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
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def reset_weights(self):
        """
        Reload the weights for the generator and discriminator Keras models
        that were previously saved before any training started, so this effectively
        resets the models.
        """
        self.generator.load_weights(self.generator_weights_file)
        self.discriminator.load_weights(self.discriminator_weights_file)

    def cleanup(self):
        """
        Perform cleanup of files before terminating the program.
        """
        try:
            os.remove(self.generator_weights_file)
        except OSError:
            # Not a big deal if these files fail to be deleted. They'll
            # get overwritten anyway during the next training run.
            print("Error while deleting generator weights file.")
        try:
            os.remove(self.discriminator_weights_file)
        except OSError:
            print("Error while deleting discriminator weights file.")

    def get_datasets(self, fold_num=0):
        """ ([int]) -> None
        For k-folds cross-validation, the current fold number (index) we're
        training with corresponds to a particular split of data into training
        and testing; this method loads those datasets.
        """
        training_datasets, testing_datasets = train_and_test_sets(fold_num)
        self.train_dataset = get_dataset(self.all_data_path, dataset_list=training_datasets)
        self.test_dataset = get_dataset(self.all_data_path, dataset_list=testing_datasets)

    def save_checkpoint(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore_from_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
