from os import listdir
from os.path import join

import PIL.ImageOps
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.image import imsave

from configs.parameters import IMAGE_DIM


class ImageIO:
    @staticmethod
    def save_enface_image(enface, filepath, filename):
        imsave(join(filepath, filename), enface, format="png", cmap="gray")

    @staticmethod
    def __load_single_image(filename, contrast_factor=1.0, sharpness_factor=1.0):
        original_image = Image.open(filename)
        contrast_enhancer = ImageEnhance.Contrast(original_image)
        contrast_image = contrast_enhancer.enhance(contrast_factor)
        sharpness_enhancer = ImageEnhance.Sharpness(contrast_image)
        return sharpness_enhancer.enhance(sharpness_factor)

    def __invert_color_scheme(self, filename, contrast_factor, sharpness_factor):
        return PIL.ImageOps.invert(self.__load_single_image(filename, contrast_factor, sharpness_factor))

    def load_single_eye(self, src_dir, contrast_factor=1.0, sharpness_factor=1.0) -> np.ndarray:
        num_images = len(listdir(src_dir))
        if num_images == 0:
            raise ValueError('FoundZeroImages')
        print('Loading {} images from `{}` ...'.format(num_images, src_dir))

        eye = np.ndarray(shape=(IMAGE_DIM, IMAGE_DIM, num_images), dtype=float)
        j = 0
        for i in range(num_images):
            try:
                j += 1
                img = self.__invert_color_scheme(join(src_dir, '{}.png'.format(j)), contrast_factor, sharpness_factor)
                eye[:, :, i] = np.asarray(img)
            except FileNotFoundError:
                i -= 1
        return eye
