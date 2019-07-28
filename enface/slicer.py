import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from skimage.transform import resize


class Slicer:
    class Orientation(Enum):
        TOP = 1
        SIDE = 2
        DEPTH = 3

    def __init__(self, eye, image_dimensions):
        self.eye = eye
        self.image_dimensions = image_dimensions

    def fly_through(self, eye, slice_indices, nrows=1, ncols=7):
        if (nrows * ncols != len(slice_indices)) or (len(slice_indices) == 0):
            raise ValueError('InvalidDimensions')
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        for row in range(nrows):
            for col in range(ncols):
                eye_slice = resize(eye[slice_indices[row * col], :, :], self.image_dimensions, anti_aliasing=True)
                axes[row][col].imshow(eye_slice, cmap='gray')
                axes[row][col].axis('off')
        plt.show()

    def multi_slice_sum(self, eye, lower, upper):
        _, y, z = eye.shape
        layers = np.ndarray(shape=(upper - lower, y, z), dtype=float)
        count = 0
        for level in range(lower, upper):
            layers[count, :, :] = eye[level, :, :]
            count += 1

        eye_summed = np.sum(layers, 0)
        resized_img = resize(eye_summed, self.image_dimensions, anti_aliasing=True)
        plt.imshow(resized_img, cmap='gray')
        plt.show()

    def multi_slice_min_norm(self, eye, lower, upper):
        (_, y, z) = eye.shape
        layers = np.ndarray(shape=(upper - lower, y, z), dtype=float)
        count = 0
        for level in range(lower, upper):
            layers[count, :, :] = eye[level, :, :]
            count += 1
        max_val = np.max(layers)
        eye_norm = np.min(np.divide(layers, max_val), 0)
        resized_img = resize(eye_norm, self.image_dimensions, anti_aliasing=True)
        plt.imshow(resized_img, cmap='gray')
        plt.show()

    def single_slice(self, eye, level, orientation: Orientation):
        if orientation == self.Orientation.DEPTH:
            eye_slice = eye[level, :, :]
        elif orientation == self.Orientation.TOP:
            eye_slice = eye[:, level, :]
        elif orientation == self.Orientation.SIDE:
            eye_slice = eye[:, :, level]
        else:
            return ValueError('InvalidOrientation')
        plt.imshow(eye_slice, cmap='gray')
        plt.show()
