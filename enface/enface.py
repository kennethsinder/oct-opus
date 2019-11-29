import sys
from glob import glob
from os import makedirs, chdir
from os.path import isfile, join, basename, normpath

from enface.image_io import ImageIO
from enface.slicer import Slicer

from configs.parameters import START_ROW, END_ROW
from datasets.train_and_test import TESTING_DATASETS


def gen_single_enface(input_dir, input_type: ImageIO.InputType, output_dir):
    image_io = ImageIO(input_dir, input_type)
    eye = image_io.load_single_eye()
    slicer = Slicer()

    multi_slice_max_norm = slicer.multi_slice_max_norm(eye=eye, lower=START_ROW, upper=END_ROW)
    image_io.save_enface_image(enface=multi_slice_max_norm, filepath=output_dir, filename="multi_slice_max_norm.png")

    multi_slice_sum = slicer.multi_slice_sum(eye=eye, lower=START_ROW, upper=END_ROW)
    image_io.save_enface_image(enface=multi_slice_sum, filepath=output_dir, filename="multi_slice_sum.png")


def gen_enface_all_testing(predicted_dir):
    for test_dataset in TESTING_DATASETS:
        gen_single_enface(
            input_dir=join(predicted_dir, test_dataset),
            input_type=ImageIO.InputType.OMAG,
            output_dir=join(predicted_dir, test_dataset)
        )


def bulk_enface():
    for folder in glob(join(sys.argv[1], '*')):
        if isfile(folder):
            continue
        folder = basename(normpath(folder))
        full_folder_path = join(sys.argv[1], folder)
        for subfolder in glob(join(full_folder_path, '*')):
            if isfile(subfolder):
                continue
            subfolder = basename(normpath(subfolder))
            # At this point `subfolder` is either "OMAG Bscans",
            # "xzOMAGInt", or "OMAG Bscans" i.e. the subfolders within
            # each eye data set.
            full_subfolder_path = join(full_folder_path, subfolder)
            makedirs(subfolder, exist_ok=True)
            chdir(subfolder)
            try:
                # Call the enface script
                # main([None, full_subfolder_path, folder])
                # TODO: call enface function
                pass
            except IndexError:
                # Error from enface script: No files in the directory
                continue
            chdir('..')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("python enface.py input_dir img_type={omag,bscan} output_dir")
        raise Exception("MissingArgument")

    if str(sys.argv[2]).lower() == "omag":
        img_type = ImageIO.InputType.OMAG
    elif str(sys.argv[2]).lower() == "bscan":
        img_type = ImageIO.InputType.BSCAN
    else:
        print("python enface.py input_dir img_type={omag,bscan} output_dir")
        raise Exception("InvalidImageType")

    gen_single_enface(input_dir=sys.argv[1], input_type=img_type, output_dir=sys.argv[3])
