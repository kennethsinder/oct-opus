from os import makedirs
from os.path import basename, join

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models, losses

import cnn.utils as utils
from cnn.parameters import IMAGE_DIM, BATCH_SIZE, SLICE_WIDTH

from datasets.train_and_test import TRAINING_DATASETS, TESTING_DATASETS


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
    def __init__(self, data_dir, checkpoints_dir):
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

        # set up checkpoints
        self.checkpoints_dir = checkpoints_dir
        self.checkpoint = tf.train.Checkpoint(model=self.model, epoch=self.epoch)
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

        # load the training and testing data
        self.data_dir = data_dir
        print('Loading training data')
        bscan_paths = utils.get_bscan_paths(self.data_dir, TRAINING_DATASETS)
        self.training_data, self.training_num_batches = utils.load_dataset(
            bscan_paths,
            BATCH_SIZE
        )
        print('Loading testing data')
        bscan_paths = utils.get_bscan_paths(self.data_dir, TESTING_DATASETS)
        self.testing_data, self.testing_num_batches = utils.load_dataset(
            bscan_paths,
            BATCH_SIZE
        )

    def train_one_epoch(self):
        """ () -> float
        Trains the model for one epoch. After the epoch is finished,
        the training data is shuffled. Returns the loss over all the
        training samples.
        """
        history = self.model.fit(
            self.training_data,
            epochs=1,
            steps_per_epoch=self.training_num_batches,
            #use_multiprocessing=True,
            #workers=16
        )
        self.epoch.assign_add(1)
        self.training_data = utils.shuffle(self.training_data, BATCH_SIZE)
        return history.history['loss']

    def test(self):
        """ () -> float
        Tests the performance of the model. Returns the loss over
        all the testing samples.
        """
        return self.model.evaluate(
            self.testing_data,
            steps=self.testing_num_batches,
            #use_multiprocessing=True,
            #workers=16
        )

    def predict(self, input_path, output_path):
        """ (dict) -> None
        Predicts the corresponding omag for given bscan.
        Saves the predictions in output_path.
        """
        dataset, num_batches = utils.load_dataset([input_path], 1, shuffle=False)
        
        predicted_slices = self.model.predict(dataset, steps=num_batches)
        if self.restore_status:
            self.restore_status.expect_partial()

        # format image
        image = utils.connect_slices(predicted_slices)
        image = np.reshape(image, [image.shape[0], image.shape[1]])
        image *= 255 # scale image from [0,1] to [0, 255]

        # save image
        Image.fromarray(image).convert('RGB').save(output_path)

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
