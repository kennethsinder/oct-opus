import sys
from os.path import isdir, join

from cgan.dataset import Dataset
from cgan.parameters import OMAG_DIRNAME, BSCAN_DIRNAME
from enface.enface import gen_single_enface

"""
Generates enface images from cross-sectional scans (e.g. B-Scans or OMAGs).
Resulting enface images are saved in the directory the images were sourced from.
"""

if __name__ == '__main__':
    if len(sys.argv) not in {3, 4} or (len(sys.argv) == 4 and sys.argv[3] != '-n') or\
            (sys.argv[1] not in {'single', 'multi'}):
        print('Usage: python enface.py <single | multi> path-to-images [-n]')
        exit(1)

    normalize = (len(sys.argv) == 4 and sys.argv[3] == '-n')

    mode = sys.argv[1]
    data_path = sys.argv[2]

    if mode == "single":
        gen_single_enface(data_path, normalize=normalize)
    elif mode == "multi":
        ds = Dataset(root_data_path=data_path)
        for dataset_name in ds.get_all_datasets():
            if isdir(join(data_path, dataset_name, BSCAN_DIRNAME)):
                gen_single_enface(join(data_path, dataset_name, BSCAN_DIRNAME), normalize=normalize)
            if isdir(join(data_path, dataset_name, OMAG_DIRNAME)):
                gen_single_enface(join(data_path, dataset_name, OMAG_DIRNAME), normalize=normalize)
            elif not isdir(join(data_path, dataset_name, BSCAN_DIRNAME)):
                # In this case, the script is probably being used to (re-)generate
                # en-faces for an `experiment-...` directory, which contains predicted
                # cross-sections in *immediate* subdirectories for each `dataset_name`.
                try:
                    gen_single_enface(join(data_path, dataset_name), normalize=normalize)
                except ValueError as e:
                    # We may be undesirably including folders in the experiment like
                    # `training_checkpoints` and `logs` which don't contain any image data,
                    # so instead of crashing, continue so we can generate the rest of the enfaces.
                    print(e)    # "FoundZeroImages"
