import glob
import random
from datetime import datetime
from os import makedirs
from os.path import basename, join

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, callbacks, Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    Dropout,
    concatenate,
    UpSampling2D
)

import cnn.utils as utils
from cnn.callback import EpochEndCallback
from cnn.parameters import IMAGE_DIM

"""
Original u-net code provided by Dr. Aaron Lee. Has been updated to work with
tensorflow.keras.
"""
def get_unet(input_height, input_width):
    inputs = Input((1, input_height, input_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    #conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    #conv1 = Dropout(0.1)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    #conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    #conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    #conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    #conv3 = Dropout(0.3)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Dropout(0.3)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.4)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Dropout(0.4)(conv5)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Dropout(0.3)(conv6)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    #conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    #conv7 = Dropout(0.3)(conv7)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    #conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    #conv8 = Dropout(0.2)(conv8)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    #conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    #conv9 = Dropout(0.1)(conv9)
    conv10 = Conv2D(1, (1, 1), activation='linear')(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    return model


class CNN:
    def __init__(self, root_data_dir, split, batch_size, slices, seed, experiment_dir):
        utils.log(
            'Creating CNN with parameters:\n' +
            '\troot_data_dir={},\n'.format(root_data_dir) +
            '\tsplit={},\n'.format(split) +
            '\tbatch_size={},\n'.format(batch_size) +
            '\tslices={},\n'.format(slices) +
            '\tseed={},\n'.format(seed) +
            '\texperiment_dir={}'.format(experiment_dir)
        )

        self.experiment_dir = experiment_dir
        self.checkpoints_dir = join(self.experiment_dir, 'checkpoints')
        self.log_dir = join(
            self.experiment_dir,
            'logs-{}'.format(datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss'))
        )
        self.cross_sections_dir = join(self.experiment_dir, 'cross_sections')
        self.enfaces_dir = join(self.experiment_dir, 'enfaces')

        makedirs(self.cross_sections_dir, exist_ok=True)
        makedirs(self.enfaces_dir, exist_ok=True)

        self.writer = tf.summary.create_file_writer(self.log_dir)

        # build model
        self.model = get_unet(IMAGE_DIM, IMAGE_DIM // slices)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.model.compile(
            optimizer=self.optimizer,
            loss=losses.MeanSquaredError()
        )

        # set up checkpoints
        self.epoch = tf.Variable(0)
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epoch
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.checkpoints_dir,
            max_to_keep=None #keep all checkpoints
        )

        if self.manager.latest_checkpoint:
            utils.log('Loading latest checkpoint {}'.format(self.manager.latest_checkpoint))
            self.restore_status = self.checkpoint.restore(self.manager.latest_checkpoint)
        else:
            self.restore_status = None

        # split data into training and testing sets
        data_names = []
        for data_dir in glob.glob(join(root_data_dir, '*')):
            data_names.append(basename(data_dir))
        data_names.sort()
        random.seed(seed)
        self.training_data_names = random.sample(data_names, int(split * len(data_names)))
        self.testing_data_names = [d for d in data_names if d not in self.training_data_names]

        # load the training and testing data
        self.training_bscan_paths = utils.get_bscan_paths(root_data_dir, self.training_data_names)

        utils.log('Calculating training data mean')
        self.training_mean = utils.get_mean(self.training_bscan_paths)
        utils.log('Done calculating training data mean')

        utils.log('Calculating training data standard deviation')
        self.training_std = utils.get_standard_deviation(
            self.training_bscan_paths, self.training_mean)
        utils.log('Done calculating training data standard deviation')

        self.training_dataset, self.training_num_batches = utils.load_dataset(
            self.training_bscan_paths,
            batch_size,
            slices,
            self.training_mean,
            self.training_std
        )

        self.testing_bscan_paths = utils.get_bscan_paths(root_data_dir, self.testing_data_names)
        self.testing_dataset, self.testing_num_batches = utils.load_dataset(
            self.testing_bscan_paths,
            batch_size,
            slices,
            self.training_mean,
            self.training_std
        )

        self.root_data_dir = root_data_dir
        self.slices = slices

    def train(self, num_epochs):
        """ (num) -> float
        Trains the model for the specified number of epochs.
        Return history
        """
        # logs loss for every batch and the average loss per epoch
        # indexing for batch loss starts at 0, while indexing for epoch loss seems to start at 1
        # also, the epoch_loss indexing is broken: it uses batch num instead of epoch num
        # ( https://github.com/tensorflow/tensorflow/issues/26873#event-2224617144 )
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=self.log_dir,
            update_freq='batch',
            profile_batch=0     # need to disable profile batching: https://github.com/tensorflow/tensorflow/issues/34276
        )
        epoch_end_callback = EpochEndCallback(self, self.epoch.numpy() + num_epochs)

        history = self.model.fit(
            self.training_dataset.repeat(num_epochs),
            steps_per_epoch=self.training_num_batches,
            initial_epoch=self.epoch.numpy(),
            epochs=self.epoch.numpy() + num_epochs,
            validation_data=self.testing_dataset,
            validation_steps=self.testing_num_batches,
            verbose=2,
            callbacks=[
                tensorboard_callback,
                epoch_end_callback
            ]
            #use_multiprocessing=True,
            #workers=16
        )
        return history

    def predict(self, input, num_batches):
        """ (str, tf.data.Dataset, int) -> numpy.ndarray
        Returns predicted image. Image has shape [C,H,W].
        """

        predicted_slices = self.model.predict(input, steps=num_batches)
        if self.restore_status:
            self.restore_status.expect_partial()

        return np.concatenate(predicted_slices, axis=2)

    def summary(self):
        """ () -> None
        Prints summary of the model
        """
        self.model.summary()

    def save_checkpoint(self):
        """ () -> None
        Saves current model state in a checkpoint.
        """
        self.manager.save()
