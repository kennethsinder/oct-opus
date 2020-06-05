import sys
from os import makedirs
from os.path import join

import numpy as np
from PIL import Image
from matplotlib.image import imsave

from cgan.dataset import Dataset
from cgan.parameters import BSCAN_DIRNAME, OMAG_DIRNAME


def flatten_single_image(image_path):
    # constants, for now
    a2 = 0.0029688
    a1 = -1.52
    a0 = 49.564

    # dimensions
    dimensions = 512

    # load images
    original = np.asarray(Image.open(image_path), dtype=int)
    flattened = np.ndarray(shape=(dimensions, dimensions), dtype=int)

    # apply rotation
    for x in range(0, dimensions):
        y = a2 * x * x + a1 * x + a0
        flattened[:, x] = np.roll(original[:, x], shift=round(y))
    return flattened


if __name__ == '__main__':
    try:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        ds = Dataset(root_data_path=input_path)
        for dataset_name in ds.get_all_datasets():
            for image_type in {BSCAN_DIRNAME, OMAG_DIRNAME}:
                makedirs(join(output_path, dataset_name, image_type), exist_ok=True)
                for image_id in range(1, 513):
                    image = flatten_single_image(join(input_path, dataset_name, image_type, "{}.png".format(image_id)))
                    imsave(join(output_path, dataset_name, image_type, "{}.png".format(image_id)),
                           image, format="png", cmap="gray")
                print("Flattened images under {}".format(join(dataset_name, image_type)))
            print("Dataset {} flattened.".format(dataset_name))
    except IndexError:
        print('Usage: python flatten.py {input_path} {output_path}')
        exit(1)
