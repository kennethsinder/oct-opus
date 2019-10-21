"""
Displays C-scan images computed from a folder supplied on standard input.
Script assumes all images in the folder have the same square dimensions.
"""

from PIL import Image
from os import listdir
from os.path import join
from enface.slicer import Slicer
from enface.loader import Loader
import matplotlib.pyplot as plt


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

    # parameters
    LOW_BOUND_LAYER = 60
    HIGH_BOUND_LAYER = 240
    IMAGE_DIMENSIONS = image_dimensions(join(src_dir, listdir(src_dir)[0]))
    print("Image Dimensions are " + str(IMAGE_DIMENSIONS))

    # loading
    loader = Loader(src_dir, input_type, IMAGE_DIMENSIONS)
    eye = loader.load_data_set()
    print("Loading Complete")

    plt.gray()  # set to gray scale

    # slices
    slicer = Slicer(eye, IMAGE_DIMENSIONS)
    multi_slice_sum = slicer.multi_slice_sum(eye, LOW_BOUND_LAYER, HIGH_BOUND_LAYER)
    plt.imsave("multi_slice_sum.png", multi_slice_sum)
    print("1/2: Multi-Slice Sum Complete (multi_slice_sum.png)")

    multi_slice_max_norm = slicer.multi_slice_max_norm(eye, LOW_BOUND_LAYER, HIGH_BOUND_LAYER)
    plt.imsave("multi_slice_max_norm.png", multi_slice_sum)
    print("2/2: Multi-Slice Max Norm Complete (multi_slice_max_norm.png)")
