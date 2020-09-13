# September 2020 - cnn/callback.py
#
# This code is an implemention of the convolutional neural network (CNN) approach
# of Dr. Aaron Y. Lee and their University of Washington team:
#
#     Lee, C.S., Tyring, A.J., Wu, Y., et al.
#     “Generating Retinal Flow Maps from Structural Optical Coherence Tomography with Artificial Intelligence,”
#     Scientific Reports 9, 5694 (2019).
#
# The paper can be found at: https://doi.org/10.1038/s41598-019-42042-y
# Credit goes to them, and we also thank them for helping us implement the model from their paper
# and particularly for their patience helping us reproduce the smaller details of their architecture.

import traceback
from os.path import join
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

import cnn.utils as utils
import cnn.image as image
from cnn.enface import generate_enface
from cnn.parameters import AUGMENT_NORMALIZE


class EpochEndCallback(callbacks.Callback):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn
        if cnn.augment_level == AUGMENT_NORMALIZE:
            self.dataset, self.num_batches = utils.load_dataset(
                [self.cnn.testing_bscan_paths[0]],
                batch_size=1,
                num_slices=cnn.slices,
                mean=cnn.mean,
                standard_deviation=cnn.std,
                shuffle=False
            )
        else:
            self.dataset, self.num_batches = utils.load_augmented_dataset(
                [self.cnn.testing_bscan_paths[0]],
                batch_size=1,
                num_slices=cnn.slices,
                use_random_jitter=cnn.use_random_jitter,
                use_random_noise=False,
                shuffle=False
            )
        self.input = []
        self.target = []
        for inp, tar in self.dataset:
            self.input.append(inp)
            self.target.append(tar)
        self.input = np.concatenate(self.input, axis=3)
        self.target = np.concatenate(self.target, axis=3)

        plt.figure(figsize=(15, 5))

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time()
        utils.log('Begin epoch')

    def on_epoch_end(self, epoch, logs=None):
        end = time()
        utils.log('End epoch - took {}s'.format(end - self.start))
        self.cnn.epoch.assign_add(1)
        self.cnn.manager.save()
        utils.log('Generating cross sections')
        self.plot_cross_sections()

    def plot_cross_sections(self):
        prediction = self.cnn.predict(self.dataset, self.num_batches)

        display_list = [self.input, self.target, prediction]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        # TODO: plot the raw input image too?
        # i.e. the image without subtracting mean and dividing by std
        plt.clf()
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            plt.imshow(tf.squeeze(display_list[i]), cmap='gray')
            plt.axis('off')

        figure_path = join(
            self.cnn.cross_sections_dir,
            'epoch_{}.png'.format(self.cnn.epoch.numpy())
        )
        plt.savefig(figure_path)

        img = image.load(figure_path)
        reshaped_img = tf.reshape(
            img, [1, img.shape[0], img.shape[1], img.shape[2]])
        with self.cnn.writer.as_default():
            tf.summary.image(
                'cross_section',
                reshaped_img,
                step=self.cnn.epoch.numpy()
            )
