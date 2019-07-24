import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from os.path import join


def fly_through(eye, nrows=1, ncols=7):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    slice_indices = range(60, 100, 1)
    for index in range(len(slice_indices)):
        eye_slice = eye[:, slice_indices[index], :]
        axes[index].imshow(eye_slice, cmap='gray')
        axes[index].axis('off')
    plt.show()


def single_slice(eye):
    fig, ax = plt.figure()
    eye_slice = eye[:, 30, :]
    ax.imshow(eye_slice, cmap='gray')
    ax.axis('off')
    plt.show()


def load_data_set(src_dir, num_images):
    eye = np.ndarray(shape=(num_images, 512, 512), dtype=float)
    for i in range(num_images):
        eye[i] = imread(join(src_dir, str(i + 1) + '.png'))
    return eye


if __name__ == '__main__':
    src = "../../training-data/2015-08-11-Images-50/xzIntensity"
    eye = load_data_set(src, 1280)
    fly_through(eye, 1, )



