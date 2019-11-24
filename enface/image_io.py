from enum import Enum
from os import listdir
from os.path import join

import PIL.ImageOps
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.image import imsave

from src.parameters import IMAGE_DIM


class ImageIO:
    class InputType(Enum):
        OMAG = 1
        BSCAN = 2

    def __init__(self, src_dir, input_type: InputType):
        self.src_dir = src_dir
        self.input_type = input_type

    @staticmethod
    def save_enface_image(enface, filepath, filename):
        imsave(join(filepath, '{}.png'.format(filename)), enface, format="png", cmap="gray")

    @staticmethod
    def __load_single_image(filename, contrast_factor=1.0, sharpness_factor=1.0):
        original_image = Image.open(filename)
        contrast_enhancer = ImageEnhance.Contrast(original_image)
        contrast_image = contrast_enhancer.enhance(contrast_factor)
        sharpness_enhancer = ImageEnhance.Sharpness(contrast_image)
        return sharpness_enhancer.enhance(sharpness_factor)

    def __invert_color_scheme(self, filename, contrast_factor, sharpness_factor):
        return PIL.ImageOps.invert(self.__load_single_image(filename, contrast_factor, sharpness_factor))

    def load_single_eye(self, contrast_factor=1.0, sharpness_factor=1.0) -> np.ndarray:
        num_images = len(listdir(self.src_dir))
        if num_images == 0:
            raise ValueError('FoundZeroImages')
        print('Loading {} images from `{}` ...'.format(num_images, self.src_dir))

        eye = np.ndarray(shape=(IMAGE_DIM, IMAGE_DIM, num_images), dtype=float)
        j = 0
        for i in range(num_images):
            try:
                j += 1
                if self.input_type == self.InputType.BSCAN:
                    img = self.__invert_color_scheme(join(self.src_dir, '{}.png'.format(j)), contrast_factor, sharpness_factor)
                else:
                    img = self.__load_single_image(join(self.src_dir, '{}.png'.format(j)), contrast_factor, sharpness_factor)
                eye[:, :, i] = np.asarray(img)
            except FileNotFoundError:
                i -= 1
        return eye
