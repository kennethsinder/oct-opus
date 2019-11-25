import os
import tensorflow as tf
from src.generator import generator
from src.discriminator import discriminator
from configs.parameters import OUTPUT_CHANNELS, ALL_DATA_DIR
from src.utils import get_dataset
from datasets.train_and_test import TRAINING_DATASETS, TESTING_DATASETS


class ModelState:

    def __init__(self):
        self.discriminator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)
        self.generator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)
        self.generator = generator(OUTPUT_CHANNELS)
        self.discriminator = discriminator()
        self.test_dataset = get_dataset(ALL_DATA_DIR, dataset_list=TESTING_DATASETS)
        self.train_dataset = get_dataset(ALL_DATA_DIR, dataset_list=TRAINING_DATASETS)
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    def save_checkpoint(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore_from_checkpoint(self):
        # cleanup old checkpoints to reduce memory footprint
        self.__moving_window_checkpoint_cleanup()
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

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
