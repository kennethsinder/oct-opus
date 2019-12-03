import sys
from glob import glob
from os import makedirs, chdir
from os.path import isfile, join, basename, normpath

from enface.enface import gen_single_enface
from enface.image_io import ImageIO

if __name__ == '__main__':
    if len(sys.argv) == 3:
        output_dir = sys.argv[1]
    elif len(sys.argv) == 4:
        output_dir = sys.argv[3]
    else:
        print("usage: python enface.py input_dir img_type={omag,bscan} output_dir=[default=input_dir]")
        raise Exception("MissingArgument")

    if str(sys.argv[2]).lower() == "omag":
        img_type = ImageIO.InputType.OMAG
    elif str(sys.argv[2]).lower() == "bscan":
        img_type = ImageIO.InputType.BSCAN
    else:
        print("usage: python enface.py input_dir img_type={omag,bscan} output_dir=[default=input_dir]")
        raise Exception("InvalidImageType")

    gen_single_enface(input_dir=sys.argv[1], input_type=img_type, output_dir=output_dir)


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
