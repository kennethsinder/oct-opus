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

    def invert_color_scheme(self, filename, enable_enhancement):
        return PIL.ImageOps.invert(self.load_single_image(filename, enable_enhancement))

    def load_single_image(self, filename, enable_enhancement):
        original_image = Image.open(filename)
        if not enable_enhancement:
            return original_image
        contrast_enhancer = ImageEnhance.Contrast(original_image)
        contrast_image = contrast_enhancer.enhance(1.5)
        sharpness_enhancer = ImageEnhance.Sharpness(contrast_image)
        return sharpness_enhancer.enhance(2.0)

    def load_data_set(self, enable_enhancement=False):
        num_images = len(listdir(self.src_dir))
        if num_images == 0:
            raise ValueError('FoundZeroImages')
        print('Loading {} images from `{}` ...'.format(num_images, self.src_dir))

        eye = np.ndarray(shape=(self.image_dimensions[1], self.image_dimensions[0], num_images), dtype=float)
        j = 0
        for i in range(num_images):
            try:
                j += 1
                if self.input_type == self.InputType.BSCAN:
                    img = self.invert_color_scheme(join(self.src_dir, '{}.png'.format(j)), enable_enhancement)
                else:
                    img = self.load_single_image(join(self.src_dir, '{}.png'.format(j)), enable_enhancement)
                eye[:, :, i] = np.asarray(img)
            except FileNotFoundError:
                i -= 1
        return eye
