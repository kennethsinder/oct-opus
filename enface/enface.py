import sys
from glob import glob
from os import makedirs, chdir
from os.path import isfile, join, basename, normpath


def gen_single_enface(dirname):
    pass


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
