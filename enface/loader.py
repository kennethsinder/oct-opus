import PIL.ImageOps
import numpy as np
from PIL import Image, ImageEnhance
from enum import Enum
from os import listdir
from os.path import join


class Loader:
    class InputType(Enum):
        OMAG = 1
        BSCAN = 2

    def __init__(self, src_dir, input_type: InputType, image_dimensions):
        self.src_dir = src_dir
        self.input_type = input_type
        self.image_dimensions = image_dimensions
        assert image_dimensions[0] == image_dimensions[1] == 512

    def save_enface_image(self, enface, filepath):
        # TODO: implement method
        pass

    @staticmethod
    def __load_single_image(filename, contrast_factor=1.0, sharpness_factor=1.0):
        original_image = Image.open(filename)
        contrast_enhancer = ImageEnhance.Contrast(original_image)
        contrast_image = contrast_enhancer.enhance(contrast_factor)
        sharpness_enhancer = ImageEnhance.Sharpness(contrast_image)
        return sharpness_enhancer.enhance(sharpness_factor)

    def __invert_color_scheme(self, filename, contrast_factor, sharpness_factor):
        return PIL.ImageOps.invert(self.__load_single_image(filename, contrast_factor, sharpness_factor))

    def load_single_eye(self, contrast_factor=1.0, sharpness_factor=1.0):
        num_images = len(listdir(self.src_dir))
        if num_images == 0:
            raise ValueError('FoundZeroImages')
        print('Loading {} images from `{}` ...'.format(num_images, self.src_dir))

        eye = np.ndarray(shape=(self.image_dimensions[0], self.image_dimensions[1], num_images), dtype=float)
        for i in range(num_images):
            try:
                if self.input_type == self.InputType.BSCAN:
                    img = self.__invert_color_scheme(join(self.src_dir, '{}.png'.format(j)), contrast_factor, sharpness_factor)
                else:
                    img = self.__load_single_image(join(self.src_dir, '{}.png'.format(j)), contrast_factor, sharpness_factor)
                eye[:, :, i] = np.asarray(img)
            except FileNotFoundError:
                i -= 1
        return eye
