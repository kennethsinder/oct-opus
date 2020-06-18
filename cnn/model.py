import glob
import random
from os import makedirs
from os.path import basename, join

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, callbacks

import cnn.utils as utils
from cnn.callback import EpochEndCallback
from cnn.parameters import IMAGE_DIM, DATASET_BLACKLIST, SLICE_WIDTH

def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    #conv1 = Dropout(0.1)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    #conv1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    #conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    #conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    #conv3 = Dropout(0.3)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    #conv3 = Dropout(0.3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    conv4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Dropout(0.4)(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    conv5 = Dropout(0.4)(conv5)
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 = Dropout(0.3)(conv6)
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    #conv7 = Dropout(0.3)(conv7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
    #conv7 = Dropout(0.3)(conv7)
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    #conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
    #conv8 = Dropout(0.2)(conv8)
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    #conv9 = Dropout(0.1)(conv9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    #conv9 = Dropout(0.1)(conv9)
    conv10 = Convolution2D(1, 1, 1, activation='linear')(conv9)
    model = Model(input=inputs, output=conv10)
    model.summary()

def downsample(input, filters):
    conv_1 = layers.Conv2D(filters, (3,3), activation='relu', padding='same')(input)
    conv_2 = layers.Conv2D(filters, (3,3), activation='relu', padding='same')(conv_1)
    max_pool = layers.MaxPool2D()(conv_2)
    return conv_2, max_pool


def upsample(input, concat_input, filters):
    # upsampling = layers.Conv2DTranspose(filters, (2,2), padding='same')(input)
    upsampling = layers.UpSampling2D((2,2))(input)
    upconv = layers.Conv2D(filters, (2,2), padding='same')(upsampling)
    concat = layers.Concatenate()([concat_input, upconv])
    conv_1 = layers.Conv2D(filters, (3,3), activation='relu', padding='same')(concat)
    conv_2 = layers.Conv2D(filters, (3,3), activation='relu', padding='same')(conv_1)
    return conv_2


class CNN:
    def __init__(self, root_data_dir, split, batch_size, seed, experiment_dir, timestring):
        print('Setting up directories')
        self.experiment_dir = experiment_dir
        self.checkpoints_dir = join(self.experiment_dir, 'checkpoints')
        self.log_dir = join(self.experiment_dir, 'logs-{}'.format(timestring))
        self.cross_sections_dir = join(self.experiment_dir, 'cross_sections')
        self.enfaces_dir = join(self.experiment_dir, 'enfaces')
        makedirs(self.cross_sections_dir, exist_ok=True)
        makedirs(self.enfaces_dir, exist_ok=True)

        self.writer = tf.summary.create_file_writer(self.log_dir)

        # build model layers
        print('Building model layers')
        input = layers.Input(shape=(IMAGE_DIM, SLICE_WIDTH, 1), name='input')

        concat_input_1, downsample_1 = downsample(input, 32)
        concat_input_2, downsample_2 = downsample(downsample_1, 64)
        concat_input_3, downsample_3 = downsample(downsample_2, 128)
        concat_input_4, downsample_4 = downsample(downsample_3, 256)

        conv = layers.Conv2D(512, (3,3), activation='relu', padding='same')(downsample_4)
        conv_2 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(conv)

        upsample_1 = upsample(conv_2, concat_input_4, 256)
        upsample_2 = upsample(upsample_1, concat_input_3, 128)
        upsample_3 = upsample(upsample_2, concat_input_2, 64)
        upsample_4 = upsample(upsample_3, concat_input_1, 32)

        output = layers.Conv2D(1, (1,1), activation='linear', padding='same')(upsample_4)

        self.model = tf.keras.Model(inputs=input, outputs=output)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.model.compile(
            optimizer=self.optimizer,
            loss=losses.MeanSquaredError()
        )

        # set up checkpoints
        print('Setting up checkpoints')
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
            print('Loading latest checkpoint: {}'.format(self.manager.latest_checkpoint))
            self.restore_status = self.checkpoint.restore(self.manager.latest_checkpoint)
        else:
            self.restore_status = None

        # split data into training and testing sets
        print('Splitting data into testing and training')
        random.seed(seed)
        data_dirs = []
        for data_dir in glob.glob(join(root_data_dir, '*')):
            if basename(data_dir) not in DATASET_BLACKLIST:
                data_dirs.append(data_dir)
        self.training_data_dirs = random.sample(data_dirs, int(split * len(data_dirs)))
        self.testing_data_dirs = [d for d in data_dirs if d not in self.training_data_dirs]

        # load the training and testing data
        print('Loading training data')
        self.training_bscan_paths = utils.get_bscan_paths(self.training_data_dirs)
        self.training_dataset = utils.load_dataset(
            self.training_bscan_paths,
            batch_size
        )
        print('Loading testing data')
        self.testing_bscan_paths = utils.get_bscan_paths(self.testing_data_dirs)
        self.testing_dataset = utils.load_dataset(
            self.testing_bscan_paths,
            batch_size
        )

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
        epoch_end_callback = EpochEndCallback(self)

        history = self.model.fit(
            self.training_dataset,
            initial_epoch=self.epoch.numpy(),
            epochs=self.epoch.numpy() + num_epochs,
            validation_data=self.testing_dataset,
            verbose=2,
            callbacks=[
                tensorboard_callback,
                epoch_end_callback
            ]
            #use_multiprocessing=True,
            #workers=16
        )
        return history

    def predict(self, input):
        """ (str, tf.data.Dataset) -> numpy.ndarray
        """

        predicted_slices = self.model.predict(input)
        if self.restore_status:
            self.restore_status.expect_partial()

        return np.concatenate(predicted_slices, axis=1)

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
