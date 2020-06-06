import sys
from os import makedirs
from os.path import join

import cv2
import numpy as np
from PIL import Image
from matplotlib.image import imsave
from scipy.ndimage import gaussian_filter

from cgan.dataset import Dataset
from cgan.parameters import BSCAN_DIRNAME, OMAG_DIRNAME

DIMENSIONS = 512


def flatten_single_image(image_path, coefficients):
    # constants, for now
    a2 = coefficients[0]
    a1 = coefficients[1]
    a0 = coefficients[2]

    # load images
    original = np.asarray(Image.open(image_path), dtype=int)
    flattened = np.ndarray(shape=(DIMENSIONS, DIMENSIONS), dtype=int)

    # apply rotation
    for x in range(0, DIMENSIONS):
        y = a2 * x * x + a1 * x + a0
        flattened[:, x] = np.roll(original[:, x], shift=round(y))
        for i in range(350, DIMENSIONS):
            flattened[i, x] = 255
    return flattened


def fit_polynomial(image_path):
    img = cv2.imread(image_path, 0)             # 0 == cv2.IMREAD_GRAYSCALE
    filtered = gaussian_filter(img, sigma=1)    # apply gaussian filter
    edges = cv2.Canny(filtered, 150, 200)       # use Canny Edge Detection

    y_values = []
    for row in range(DIMENSIONS):
        found = False
        for col in range(DIMENSIONS):
            if edges[row][col] > 0:
                y_values.append(col)
                found = True
                break
        if not found:
            if row > 0:
                y_values.append(y_values[row-1])
            else:
                y_values.append(0)

    assert len(y_values) == DIMENSIONS

    x = np.arange(DIMENSIONS)
    y = np.array(y_values)
    return np.polyfit(x, y, 2)


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
