import gc
import os
import tensorflow as tf
from src.generator import generator
from src.discriminator import discriminator
from src.parameters import OUTPUT_CHANNELS, TEST_DATA_DIR, TRAIN_DATA_DIR
from src.utils import get_dataset


class ModelState:

    def __init__(self):
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.initialize()

    def initialize(self):
        self.train_dataset = get_dataset(TRAIN_DATA_DIR)
        self.test_dataset = get_dataset(TEST_DATA_DIR)
        self.discriminator = discriminator()
        self.generator = generator(OUTPUT_CHANNELS)
        self.generator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            5e-4, beta_1=0.5)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def fuck_it_all(self):
        """
        Currently unused because turns out
        this does not actually free RAM (sigh).
        """
        self.train_dataset = None
        self.test_dataset = None
        self.discriminator = None
        self.generator = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        gc.collect()
        self.initialize()
        self.restore_from_checkpoint()

    def save_checkpoint(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore_from_checkpoint(self):
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))
