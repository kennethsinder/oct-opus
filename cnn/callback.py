from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

import cnn.utils as utils
import cnn.image as image
from cnn.enface import generate_enface


class EpochEndCallback(callbacks.Callback):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn
        self.dataset = utils.load_dataset(
            [self.cnn.testing_bscan_paths[0]], 1, shuffle=False)
        self.input = []
        self.target = []
        for inp, tar in self.dataset:
            self.input.append(inp)
            self.target.append(tar)
        self.input = np.concatenate(self.input, axis=3)
        self.target = np.concatenate(self.target, axis=3)

    def on_epoch_end(self, epoch, logs=None):
        self.cnn.epoch.assign_add(1)
        self.cnn.manager.save()
        self.plot_cross_sections()
        self.generate_enfaces()

    def plot_cross_sections(self):
        print('Logging cross sections')

        prediction = self.cnn.predict(self.dataset)

        display_list = [self.input, self.target, prediction]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        plt.figure(figsize=(15, 5))
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
        print('Logging enfaces')
        for testing_dir in self.cnn.testing_data_dirs:
            generate_enface(self.cnn, testing_dir)
