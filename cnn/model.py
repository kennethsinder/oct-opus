import glob
import random
from os import makedirs
from os.path import basename, isfile, join, splitext

import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks

import cnn.utils as utils
import cnn.image as image
from cnn.parameters import (
    IMAGE_DIM,
    DATASET_BLACKLIST,
    SLICE_WIDTH,
    PIXEL_DEPTH,
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
    def __init__(self, root_data_dir, split, batch_size, seed, experiment_dir):
        self.experiment_dir = experiment_dir
        self.checkpoints_dir = join(self.experiment_dir, 'checkpoints')
        self.log_dir = join(self.experiment_dir, 'log')
        self.enface_dir = join(self.experiment_dir, 'enface')

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
        self.optimizer = tf.keras.optimizers.Adam()
        self.model.compile(
            optimizer=self.optimizer,
            loss=losses.MeanAbsoluteError()
        )

        # set up checkpoints
        self.epoch = tf.Variable(0)
        self.root_data_dir = tf.Variable(root_data_dir)
        self.split = tf.Variable(split)
        self.batch_size = tf.Variable(batch_size)
        self.seed = tf.Variable(seed)

        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epoch,
            batch_size=self.batch_size,
            root_data_dir=self.root_data_dir,
            split=self.split,
            seed=self.seed,
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
        random.seed(self.seed.numpy())
        data_dirs = []
        for data_dir in glob.glob(join(self.root_data_dir.numpy().decode('utf-8'), '*')):
            if basename(data_dir) not in DATASET_BLACKLIST:
                data_dirs.append(data_dir)
        self.training_data_dirs = random.sample(data_dirs, int(self.split.numpy() * len(data_dirs)))
        self.testing_data_dirs = [d for d in data_dirs if d not in self.training_data_dirs]

        # load the training and testing data
        self.training_dataset, self.training_num_batches = utils.load_dataset(
            self.training_data_dirs,
            self.batch_size.numpy()
        )
        self.testing_dataset, self.testing_num_batches = utils.load_dataset(
            self.testing_data_dirs,
            self.batch_size.numpy()
        )

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

        class ManagerCallback(callbacks.Callback):
            def __init__(self, manager, epoch_counter):
                super().__init__()
                self.manager = manager
                self.epoch_counter = epoch_counter

            def on_epoch_end(self, epoch, logs=None):
                self.epoch_counter.assign_add(1)
                self.manager.save()

        # logs loss for every batch and the average loss per epoch
        # indexing for batch loss starts at 0, while indexing for epoch loss seems to start at 1
        # also, the epoch_loss indexing is broken: it uses batch num instead of epoch num
        # ( https://github.com/tensorflow/tensorflow/issues/26873#event-2224617144 )
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=self.log_dir,
            update_freq='batch',
            profile_batch=0     # need to disable profile batching: https://github.com/tensorflow/tensorflow/issues/34276
        )
        manager_callback = ManagerCallback(self.manager, self.epoch)

        history = self.model.fit(
            self.training_dataset.repeat(num_epochs),
            initial_epoch=self.epoch.numpy(),
            epochs=self.epoch.numpy() + num_epochs,
            steps_per_epoch=self.training_num_batches,
            validation_data=self.testing_dataset,
            validation_steps=self.testing_num_batches,
            verbose=2,
            callbacks=[
                tensorboard_callback,
                manager_callback
            ]
            #use_multiprocessing=True,
            #workers=16
        )
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
