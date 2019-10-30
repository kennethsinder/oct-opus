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
import sys


def image_dimensions(filename):
    return Image.open(filename).size


if __name__ == '__main__':
    suffix = None
    if len(sys.argv) == 1:
        # Usage: python plot.py
        #        ...and then information supplied as input to stdin.
        src_dir = str(
            input("Enter the absolute path to the images you wish to process ... : "))
        img_type = str(input("Enter the image type {OMAG|BSCAN} ... : "))
        if img_type == "OMAG":
            input_type = Loader.InputType.OMAG
        elif img_type == "BSCAN":
            input_type = Loader.InputType.BSCAN
        else:
            raise Exception("UnknownInputType")
    elif len(sys.argv) in {3, 4}:
        # Usage: python plot.py <directory path here> <suffix for file names here>
        src_dir = sys.argv[1]
        suffix = sys.argv[2]
        input_type = Loader.InputType.OMAG if sys.argv[-1] == 'OMAG' else Loader.InputType.BSCAN
    else:
        raise Exception('Invalid number of command line arguments')

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
    file_name = 'multi_slice_sum.png' if not suffix else 'multi_slice_sum_{}.png'.format(suffix)
    plt.imsave(file_name, multi_slice_sum)
    print('1/2: Multi-Slice Sum Complete ({})'.format(file_name))

    multi_slice_max_norm = slicer.multi_slice_max_norm(eye, LOW_BOUND_LAYER, HIGH_BOUND_LAYER)
    file_name = 'multi_slice_max_norm.png' if not suffix else 'multi_slice_max_norm_{}.png'.format(suffix)
    plt.imsave(file_name, multi_slice_sum)
    print('2/2: Multi-Slice Max Norm Complete ({})'.format(file_name))
