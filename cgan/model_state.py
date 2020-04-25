import os
from os.path import join

import tensorflow as tf

from configs.parameters import ALL_DATA_DIR
from datasets.train_and_test import train_and_test_sets
from cgan.discriminator import discriminator
from cgan.generator import generator
from cgan.utils import get_dataset


class ModelState:

    def __init__(self, root_data_dir):
        self.discriminator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)
        self.generator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)
        self.generator = generator()
        self.discriminator = discriminator()
        self.all_data_path = join(root_data_dir, ALL_DATA_DIR)
        self.train_dataset = None
        self.test_dataset = None
        self.checkpoint_dir = './training_checkpoints'
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
        self.generator_weights_file = 'generator_weights.h5'
        self.discriminator_weights_file = 'discriminator_weights.h5'
        self.generator.save_weights(self.generator_weights_file)
        self.discriminator.save_weights(self.discriminator_weights_file)
        self.current_training_step = 0

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
        # cleanup old checkpoints to reduce memory footprint
        self.__moving_window_checkpoint_cleanup()
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))

    @staticmethod
    def __get_epoch_num(checkpoint_name):
        return int(checkpoint_name.split(".")[0].split("-")[1])

    def __moving_window_checkpoint_cleanup(self, window_size=5):
        # list of all checkpoint files
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_files = os.listdir(self.checkpoint_dir)
        if 'checkpoint' in checkpoint_files:
            # no need to consider this file
            checkpoint_files.remove("checkpoint")

        if len(checkpoint_files) == 0:
            return

        # sort by asc epoch numbers
        checkpoint_files.sort(key=self.__get_epoch_num, reverse=False)

        # cleanup old checkpoints
        highest_epoch_num = self.__get_epoch_num(checkpoint_files[-1])
        for checkpoint in checkpoint_files:
            if self.__get_epoch_num(checkpoint) < (highest_epoch_num - window_size):
                os.remove(self.checkpoint_dir + "/" + checkpoint)
                print("cleaned up checkpoint ", checkpoint)
