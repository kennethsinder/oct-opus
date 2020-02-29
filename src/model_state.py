import os
from os.path import join

import tensorflow as tf

from configs.parameters import ALL_DATA_DIR
from datasets.train_and_test import train_and_test_sets
from src.discriminator import discriminator
from src.generator import generator
from src.utils import get_dataset


class ModelState:

    def __init__(self, root_data_dir):
        self.discriminator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)
        self.generator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)
        self.generator = generator()
        self.discriminator = discriminator()
        self.all_data_path = join(root_data_dir, ALL_DATA_DIR)
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    def get_datasets(self, fold_num=0):
        TRAINING_DATASETS, TESTING_DATASETS = train_and_test_sets(fold_num)
        self.test_dataset = get_dataset(self.all_data_path, dataset_list=TESTING_DATASETS)
        self.train_dataset = get_dataset(self.all_data_path, dataset_list=TRAINING_DATASETS)

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
