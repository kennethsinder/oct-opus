import sys
from glob import glob
from os import makedirs, chdir
from os.path import isfile, join, basename, normpath

from enface.image_io import ImageIO
from enface.slicer import Slicer

from src.parameters import START_ROW, END_ROW


def gen_single_enface(input_dir, input_type: ImageIO.InputType, output_dir, output_file):
    image_io = ImageIO(input_dir, input_type)
    eye = image_io.load_single_eye()
    slicer = Slicer()

    multi_slice_max_norm = slicer.multi_slice_max_norm(eye=eye, lower=START_ROW, upper=END_ROW)
    image_io.save_enface_image(enface=multi_slice_max_norm, filepath=output_dir, filename=output_file)

    multi_slice_sum = slicer.multi_slice_sum(eye=eye, lower=START_ROW, upper=END_ROW)
    image_io.save_enface_image(enface=multi_slice_sum, filepath=output_dir, filename=output_file)


def gen_enface_all_testing():
    pass


def bulk_enface():
    if len(sys.argv) < 2:
        raise Exception("MissingArgument")

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
