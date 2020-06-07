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


def flatten_single_image(image_path, polynomial):
    # constants, for now
    a2 = polynomial[0]
    a1 = polynomial[1]
    a0 = polynomial[2]

    # load images
    original = np.asarray(Image.open(image_path), dtype=int)
    flattened = np.ndarray(shape=(DIMENSIONS, DIMENSIONS), dtype=int)

    # apply rotation
    for x in range(0, DIMENSIONS):
        y = a2 * x * x + a1 * x + a0
        flattened[:, x] = np.roll(original[:, x], shift=int(round(y)))
        for i in range(350, DIMENSIONS):
            flattened[i, x] = 255
    return flattened


def fit_polynomial(image_path):
    img = cv2.imread(image_path, 0)             # 0 == cv2.IMREAD_GRAYSCALE
    filtered = gaussian_filter(img, sigma=1)    # apply gaussian filter
    edges = cv2.Canny(filtered, 150, 200)       # use Canny Edge Detection

    y_values = []
    high, low = DIMENSIONS - 50, 50  # disregard first/last 50 columns near edges
    for col in range(low, high):
        found = False
        for row in range(DIMENSIONS):
            if edges[row][col] > 0:
                y_values.append(-col)  # note negative value used
                found = True
                break
        if not found:
            if len(y_values) > 0:
                y_values.append(y_values[len(y_values)-1])  # use previous value if needed
            else:
                y_values.append(0)

    assert len(y_values) == high - low

    x = np.array(list(range(low, high)))
    y = np.array(y_values)
    return np.polyfit(x, y, 2)


if __name__ == '__main__':
    try:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        ds = Dataset(root_data_path=input_path)
        for dataset_name in ds.get_all_datasets():
            # Create output directories
            makedirs(join(output_path, dataset_name, OMAG_DIRNAME), exist_ok=True)
            makedirs(join(output_path, dataset_name, BSCAN_DIRNAME), exist_ok=True)

            # Loop over individual images
            for image_id in range(1, DIMENSIONS + 1):
                # Fits a polynomial to the cross section. Note that `BSCAN_DIRNAME` is always used
                poly = fit_polynomial(join(input_path, dataset_name, BSCAN_DIRNAME, "{}.png".format(image_id)))

                # alternatively, use hardcoded polynomial coefficients
                # poly = np.array([0.0029688, -1.52, 49.564])

                for image_type in {BSCAN_DIRNAME, OMAG_DIRNAME}:
                    # Flattened image
                    image = flatten_single_image(
                        image_path=join(input_path, dataset_name, image_type, "{}.png".format(image_id)),
                        polynomial=poly
                    )

                    # Saves to disk
                    imsave(
                        fname=join(output_path, dataset_name, image_type, "{}.png".format(image_id)),
                        arr=image, format="png", cmap="gray"
                    )
            print("Dataset {} flattened.".format(dataset_name))
    except IndexError:
        print('Usage: python flatten.py {input_path} {output_path}')
        exit(1)
