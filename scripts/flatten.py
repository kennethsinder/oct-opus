import sys
from glob import glob
from os.path import join

import numpy as np
from PIL import Image

from cgan.dataset import Dataset
from cgan.parameters import BSCAN_DIRNAME, OMAG_DIRNAME


def flatten_single_image(image_path):
    # constants, for now
    a2 = 0.0029688
    a1 = -1.52
    a0 = 49.564

    cross_section = np.asarray(Image.open(image_path))

    # TODO: rest of implementation

    dimensions = 512
    for x in range(0, dimensions):
        y = a2 * x * x + a1 * x + a0
        y_round = round(y)


if __name__ == '__main__':
    try:
        data_path = sys.argv[1]
        ds = Dataset(root_data_path=data_path)
        for dataset_name in ds.get_all_datasets():
            for image_type in {BSCAN_DIRNAME, OMAG_DIRNAME}:
                ls = glob(join(join(data_path, dataset_name, image_type), '[0-9]*.png'))
                assert len(ls) > 0
                for filename in ls:
                    flatten_single_image(join(data_path, dataset_name, image_type, filename))
    except IndexError:
        print('Usage: python scripts/flatten.py {path-to-images}')
        exit(1)
