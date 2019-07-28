import PIL.ImageOps
import numpy as np
from PIL import Image
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

    def invert_color_scheme(self, filename):
        return PIL.ImageOps.invert(Image.open(filename))

    def load_data_set(self):
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
                    img = self.invert_color_scheme(join(self.src_dir, '{}.png'.format(j)))
                else:
                    img = PIL.Image.open(join(self.src_dir, '{}.png'.format(j)))
                eye[:, :, i] = np.asarray(img)
            except FileNotFoundError:
                i -= 1
        return eye
