from enum import Enum
from typing import List

import numpy as np
from skimage.transform import resize

from configs.parameters import IMAGE_DIM


class Slicer:
    class Orientation(Enum):
        TOP = 1
        SIDE = 2
        DEPTH = 3

    def __init__(self):
        pass

    def fly_through(self, eye: np.ndarray, slice_indices: List[int], anti_aliasing=False) -> List[np.ndarray]:
        res = []
        for slice_index in slice_indices:
            eye_slice = resize(eye[slice_index, :, :], IMAGE_DIM, anti_aliasing=anti_aliasing)
            res.append(eye_slice)
        return res

    def multi_slice_sum(self, eye, lower, upper, anti_aliasing=False) -> np.ndarray:
        _, y, z = eye.shape
        layers = np.ndarray(shape=(upper - lower, y, z), dtype=float)
        count = 0
        for level in range(lower, upper):
            layers[count, :, :] = eye[level, :, :]
            count += 1
        eye_summed = np.sum(layers, 0)
        return resize(eye_summed, IMAGE_DIM, anti_aliasing=anti_aliasing)

    def multi_slice_max_norm(self, eye: np.ndarray, lower, upper, anti_aliasing=False) -> np.ndarray:
        (_, y, z) = eye.shape
        layers = np.ndarray(shape=(upper - lower, y, z), dtype=float)
        count = 0
        for level in range(lower, upper):
            layers[count, :, :] = eye[level, :, :]
            count += 1
        max_val = np.max(layers)
        eye_norm = np.max(np.divide(layers, max_val), 0)
        return resize(eye_norm, IMAGE_DIM, anti_aliasing=anti_aliasing)

    def single_slice(self, eye: np.ndarray, level, orientation: Orientation) -> np.ndarray:
        if orientation == self.Orientation.DEPTH:
            eye_slice = eye[level, :, :]
        elif orientation == self.Orientation.TOP:
            eye_slice = eye[:, level, :]
        else:  # orientation == self.Orientation.SIDE:
            eye_slice = eye[:, :, level]
        return eye_slice
