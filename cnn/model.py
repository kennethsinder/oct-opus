import glob
import random
from os import makedirs
from os.path import basename, isfile, join, splitext

import tensorflow as tf
from tensorflow.keras import layers, models, losses

import cnn.utils as utils
import cnn.image as image
from cnn.parameters import (
    IMAGE_DIM,
    DATASET_BLACKLIST,
    SLICE_WIDTH,
    PIXEL_DEPTH,
    ROOT_ENFACE_DIR,
)


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
    def __init__(self, root_data_dir, split, batch_size, checkpoints_dir):
        self.root_data_dir = root_data_dir
        self.split = split
        self.batch_size = batch_size
        self.checkpoints_dir = checkpoints_dir

        # build model layers
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
        self.model.compile(
            optimizer='Adam',
            loss=losses.MeanAbsoluteError()
        )

        self.epoch = tf.Variable(0)

        # split data into training and testing sets
        data_dirs = []
        for data_dir in glob.glob(join(self.root_data_dir, '*')):
            if basename(data_dir) not in DATASET_BLACKLIST:
                data_dirs.append(data_dir)
        self.training_data_dirs = random.sample(data_dirs, int(self.split * len(data_dirs)))
        self.testing_data_dirs = [d for d in data_dirs if d not in self.training_data_dirs]

        # load the training and testing data
        self.training_data, self.training_num_batches = utils.load_dataset(
            self.training_data_dirs,
            self.batch_size
        )
        self.testing_data, self.testing_num_batches = utils.load_dataset(
            self.testing_data_dirs,
            self.batch_size
        )

        # convert to tf Variables
        self.training_data_dirs = tf.Variable(self.training_data_dirs)
        self.testing_data_dirs = tf.Variable(self.testing_data_dirs)
        self.training_num_batches = tf.Variable(self.training_num_batches)
        self.testing_num_batches = tf.Variable(self.testing_num_batches)

        # set up checkpoints
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            epoch=self.epoch,
            training_data_dirs=self.training_data_dirs,
            training_data=self.training_data,
            training_num_batches=self.training_num_batches,
            testing_data_dirs=self.testing_data_dirs,
            testing_data=self.testing_data,
            testing_num_batches=self.testing_num_batches,
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


    def train(self, num_epochs):
        """ (num) -> float
        Trains the model for the specified number of epochs.
        Return history
        """
        # TODO: add callbacks to:
        #   - save checkpoints
        #   - generate enfaces
        #   - increment self.epoch
        # after every epoch
        history = self.model.fit(
            self.training_data.repeat(num_epochs),
            initial_epoch=self.epoch_num(),
            epochs=num_epochs,
            steps_per_epoch=self.training_num_batches.numpy(),
            validation_data=self.testing_data,
            validation_steps=self.testing_num_batches.numpy(),
            verbose=2,
            #use_multiprocessing=True,
            #workers=16
        )
        self.epoch.assign_add(num_epochs)
        return history

    def predict(self, input_path):
        """ (str, str) -> numpy.ndarray
        Returns the predicted omag for given bscan located in input_path.
        """
        dataset, num_batches = utils.load_dataset([input_path], 1, shuffle=False)

        # get slices
        predicted_slices = self.model.predict(dataset, steps=num_batches)
        if self.restore_status:
            self.restore_status.expect_partial()

        return image.connect(predicted_slices)

    def epoch_num(self):
        return self.epoch.numpy()

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
