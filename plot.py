"""
Displays C-scan images computed from a folder supplied on standard input.
Script assumes all images in the folder have the same square dimensions.
"""

from PIL import Image
from os import listdir
from os.path import join
from enface.slicer import Slicer
from enface.loader import Loader

LOW_BOUND_LAYER = 60
HIGH_BOUND_LAYER = 240


def image_dimensions(filename):
    return Image.open(filename).size


if __name__ == '__main__':
    src_dir = str(
        input("Enter the absolute path to the images you wish to process ... : "))
    img_type = str(input("Enter the image type {OMAG|BSCAN} ... : "))
    if img_type == "OMAG":
        input_type = Loader.InputType.OMAG
    elif img_type == "BSCAN":
        input_type = Loader.InputType.BSCAN
    else:
        raise Exception("UnknownInputType")

    IMAGE_DIMENSIONS = image_dimensions(join(src_dir, listdir(src_dir)[0]))
    print("Image Dimensions are " + str(IMAGE_DIMENSIONS))
    loader = Loader(src_dir, input_type, IMAGE_DIMENSIONS)
    eye = loader.load_data_set()
    print("Loading Complete")

    slicer = Slicer(eye, IMAGE_DIMENSIONS)
    slicer.multi_slice_sum(eye, LOW_BOUND_LAYER, HIGH_BOUND_LAYER)
    slicer.multi_slice_max_norm(eye, LOW_BOUND_LAYER, HIGH_BOUND_LAYER)
    slicer.fly_through(eye, range(LOW_BOUND_LAYER, HIGH_BOUND_LAYER,
                                  (HIGH_BOUND_LAYER - LOW_BOUND_LAYER) // 6), 2, 3)
