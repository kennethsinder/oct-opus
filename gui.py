#!/usr/bin/python

from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import *
import tkinter
import os
import cv2
import numpy as np
import os.path
import datetime
import time

from cgan.dataset import Dataset
from cgan.model_state import ModelState
from cgan.utils import generate_inferred_images

# Silence deprecation warning on Mac
os.environ['TK_SILENCE_DEPRECATION'] = '1'
 
window = tkinter.Tk()
window.title('Retinal Capillary ML Testing Tool')
window.grid_columnconfigure(0, pad=30)
window.grid_columnconfigure(1, pad=30)
window.grid_rowconfigure(0, pad=30)
window.grid_rowconfigure(1, pad=30)
window.grid_rowconfigure(2, pad=30)

global image_file_path, ckptdir
image_file_path = None
ckptdir = None


# export directory used to store output
EXP_DIR = "experiment-{}".format(
    datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))


# Button handlers
# ---------------
def choose_input_directory(event):
    global image_file_path
    image_file_path = askdirectory(
        initialdir=".", title="Select input directory")
    print(image_file_path if image_file_path else 'No input directory chosen')

def choose_checkpoint(event):
    # load from latest checkpoint and load data for just 1 of 5 folds
    global ckptdir
    ckptdir = askopenfilename(
        initialdir=".", title="Select checkpoint")
    print(ckptdir if ckptdir else 'No checkpoint chosen')

def generate(event):
    print(image_file_path)
    print(ckptdir)
    print(EXP_DIR)
    assert image_file_path is not None
    ds = Dataset(root_data_path=image_file_path, num_folds=1)
    
    os.makedirs(EXP_DIR, exist_ok=False)
    with open(os.path.join(EXP_DIR, 'README.md'), 'w') as readme_file:
        # Create a mostly blank README file to encourage good
        # documentation of the purpose of each experiment.
        readme_file.write('# {}\n\n'.format(EXP_DIR))

    assert ckptdir is not None
    model_state = ModelState(
            is_training_mode=False, ckpt_dir=ckptdir, dataset=ds)
    model_state.is_training_mode = False
    print('Restoring from checkpoint at {}'.format(ckptdir))
    model_state.restore_from_checkpoint()

    # generate results based on prediction
    generate_inferred_images(EXP_DIR, model_state)
# ---------------

# Checkpoint button/label
lbl = Label(window, text="1. Training checkpoint (ei. ML model):")
lbl.grid(column=0, row=1)

ckpt_button = tkinter.Button(
    window, text='Choose checkpoint')
ckpt_button.grid(column=1, row=1)
ckpt_button.bind('<Button-1>', choose_checkpoint)

# Input directory button/label
input_img_lbl = Label(window, text="""2. Select test input image directory:
    Note: This should be the parent directory
    for all target dataset folders.""")
input_img_lbl.grid(column=0, row=2)

button_widget = tkinter.Button(
    window, text='Choose input directory')
button_widget.grid(column=1, row=2)
button_widget.bind('<Button-1>', choose_input_directory)

# Generate button/label
gen_lbl = Label(window, text="""3. Generate and export to output directory: 
    {}""".format(EXP_DIR))
gen_lbl.grid(column=0, row=3)
gen_button = tkinter.Button(
    window, text='Generate Inferred Images')
gen_button.grid(column=1, row=3)
gen_button.bind('<Button-1>', generate)


window.mainloop()
