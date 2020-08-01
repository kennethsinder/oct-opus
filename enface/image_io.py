import cv2
import glob
from os.path import join

import PIL.ImageOps
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.image import imsave
from typing import List

# filename constants
MULTI_SLICE_MAX_NORM = "multi_slice_max_norm.png"
MULTI_SLICE_SUM = "multi_slice_sum.png"


class ImageIO:
    def __init__(self, IMAGE_DIM):
        self.__IMAGE_DIM = IMAGE_DIM

    @staticmethod
    def save_enface_image(enface, filepath, filename, normalize=False):
        imsave(join(filepath, filename), enface, format="png", cmap="gray")
        if normalize:
            # Apply histogram equalization if requested
            img = cv2.imread(join(filepath, filename), 0)
            equ = cv2.equalizeHist(img)
            cv2.imwrite(join(filepath, filename), equ)

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
        ls = glob.glob(join(src_dir, '[0-9]*.png'))
        num_images = len(ls)
        if num_images == 0:
            raise ValueError('FoundZeroImages')
        print('Loading {} images from `{}` ...'.format(num_images, src_dir))

        eye = np.ndarray(shape=(self.__IMAGE_DIM, self.__IMAGE_DIM, num_images), dtype=float)
        j = 0
        for i in range(num_images):
            try:
                j += 1
                # cross-sectional scans (e.g. B-Scans, OMAGs) should be black lines on white background
                # enfaces, on the other hand, are white lines on black background
                img = self.__invert_color_scheme(join(src_dir, '{}.png'.format(j)), contrast_factor, sharpness_factor)
                eye[:, :, i] = np.asarray(img)
            except FileNotFoundError:
                i -= 1
        return eye

    def load_single_eye_from_mem(self,
                                 cross_sections: List[Image.Image]) -> np.ndarray:
        num_images = len(cross_sections)
        eye = np.ndarray(
            shape=(self.__IMAGE_DIM, self.__IMAGE_DIM, num_images),
            dtype=float
        )
        for i, img in enumerate(cross_sections):
            eye[:, :, i] = np.asarray(PIL.ImageOps.invert(
                img
            ))
        return eye
