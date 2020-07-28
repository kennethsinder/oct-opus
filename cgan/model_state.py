import os
from os.path import join

import tensorflow as tf

from cgan.dataset import Dataset
from cgan.discriminator import discriminator
from cgan.generator import generator
from cgan.utils import get_dataset


class ModelState:

    def __init__(self, is_training_mode: bool, ckpt_dir: str, dataset: Dataset):
        self.is_training_mode = is_training_mode

        # optimizers
        self.discriminator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)
        self.generator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.5)

        # generator and discriminator
        self.generator = generator()
        self.discriminator = discriminator()

        # paths
        self.CKPT_DIR = ckpt_dir

        # dataset
        self.DATASET = dataset

        # datasets
        self.train_dataset = None
        self.test_dataset = None

        # checkpoints
        self.checkpoint_prefix = os.path.join(self.CKPT_DIR, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

        if self.is_training_mode:
            # Save initial, scrambled weights before we do any training
            # so that we can reload them with model_state.reset_weights()
            # below between folds when doing k-folds cross-validation.
            self.save_checkpoint()
            self.initial_checkpoint = tf.train.latest_checkpoint(self.CKPT_DIR)
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

        # Training index across all epochs and folds
        self.global_index = 0

    def reset_weights(self):
        """
        Reload the weights for the generator and discriminator Keras models
        that were previously saved before any training started, so this effectively
        resets the models. Also resets the generator & discriminator Adam optimizers.
        """
        self.checkpoint.restore(self.initial_checkpoint)

    def get_datasets(self, fold_num=0):
        """ ([int]) -> None
        For k-folds cross-validation, the current fold number (index) we're
        training with corresponds to a particular split of data into training
        and testing; this method loads those datasets.
        """
        training_datasets, testing_datasets = self.DATASET.get_train_and_test_by_fold_id(fold_num)
        self.train_dataset = get_dataset(self.DATASET.root_data_path, dataset_iterable=training_datasets)
        self.test_dataset = get_dataset(self.DATASET.root_data_path, dataset_iterable=testing_datasets)

        # Also store which folders we're *not* training on,
        # to use later for prediction.
        self.test_folder_names = testing_datasets

    def save_checkpoint(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore_from_checkpoint(self, override_checkpoint_dir: str = None):
        """
        Restores the model from a checkpoint file. Optionally allows the
        checkpoints folder path to be specified by passing it in as an
        argument, otherwise just uses the checkpoint dir supplied in when
        the `ModelState` was instantiated.
        """
        ckpt_dir = self.CKPT_DIR
        if override_checkpoint_dir is not None:
            ckpt_dir = override_checkpoint_dir

        self.checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
