from os.path import join
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

import cnn.utils as utils
import cnn.image as image
from cnn.enface import generate_enface


class EpochEndCallback(callbacks.Callback):
    def __init__(self, cnn, last_epoch):
        super().__init__()
        self.cnn = cnn
        self.last_epoch = last_epoch
        self.dataset, self.num_batches = utils.load_dataset(
            [self.cnn.testing_bscan_paths[0]],
            batch_size=1,
            num_slices=cnn.slices,
            mean=cnn.training_mean,
            standard_deviation=cnn.training_std,
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
        if self.cnn.epoch.numpy() % 5 == 0 or self.cnn.epoch.numpy() == self.last_epoch:
            utils.log('Generating enfaces')
            self.generate_enfaces()

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

    def generate_enfaces(self):
        for dir in self.cnn.testing_data_dirs:
            generate_enface(self.cnn, dir)
