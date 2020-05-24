import sys
from os.path import join

from cgan.dataset import Dataset
from cgan.parameters import OMAG_DIRNAME, BSCAN_DIRNAME
from enface.enface import gen_single_enface

"""
Generates enface images from cross-sectional scans (e.g. B-Scans or OMAGs).
Resulting enface images are saved in the directory the images were sourced from.
"""

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: python enface.py {single, multi} [path to images]")
        exit(1)

    mode = sys.argv[1]
    data_path = sys.argv[2]

    if mode == "single":
        gen_single_enface(data_path)
    elif mode == "all":
        ds = Dataset(root_data_path=data_path)
        for dataset_name in ds.get_all_datasets():
            for scan_type in {OMAG_DIRNAME, BSCAN_DIRNAME}:
                gen_single_enface(join(data_path, dataset_name, scan_type))
    else:
        print("usage: python enface.py {single, multi} [path to images]")
        exit(1)
