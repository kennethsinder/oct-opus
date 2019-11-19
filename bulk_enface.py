import glob
import os
import os.path
import sys
from plot import main

for folder in glob.glob(os.path.join(sys.argv[1], '*')):
    if os.path.isfile(folder):
        continue
    folder = os.path.basename(os.path.normpath(folder))
    full_folder_path = os.path.join(sys.argv[1], folder)
    for subfolder in glob.glob(os.path.join(full_folder_path, '*')):
        if os.path.isfile(subfolder):
            continue
        subfolder = os.path.basename(os.path.normpath(subfolder))
        # At this point `subfolder` is either "OMAG Bscans",
        # "xzOMAGInt", or "OMAG Bscans" i.e. the subfolders within
        # each eye data set.
        full_subfolder_path = os.path.join(full_folder_path, subfolder)
        os.makedirs(subfolder, exist_ok=True)
        os.chdir(subfolder)
        try:
            # Call the enface script
            main([None, full_subfolder_path, folder])
        except IndexError:
            # Error from enface script: No files in the directory
            continue
        os.chdir('..')
