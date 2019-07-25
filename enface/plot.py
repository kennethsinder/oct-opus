import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from os.path import join
from skimage.transform import resize
from os import listdir


def fly_through(eye, slice_indices, nrows=1, ncols=7):
    if (nrows * ncols != len(slice_indices)) or (len(slice_indices) == 0):
        raise ValueError('InvalidDimensions')
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    for row in range(nrows):
        for col in range(ncols):
            eye_slice = resize(eye[slice_indices[row * col], :, :], (512, 512), anti_aliasing=True)
            axes[row][col].imshow(eye_slice, cmap='gray')
            axes[row][col].axis('off')
    plt.show()


def multi_slice_sum(eye, lower, upper):
    _, y, z = eye.shape
    layers = np.ndarray(shape=(upper - lower, y, z), dtype=float)
    count = 0
    for level in range(lower, upper):
        layers[count, :, :] = eye[level, :, :]
        count += 1

    eye_summed = np.sum(layers, 0)
    resized_img = resize(eye_summed, (512, 512), anti_aliasing=True)
    plt.imshow(resized_img, cmap='gray')
    plt.show()


def multi_slice_min_norm(eye, lower, upper):
    (_, y, z) = eye.shape
    layers = np.ndarray(shape=(upper - lower, y, z), dtype=float)
    count = 0
    for level in range(lower, upper):
        layers[count, :, :] = eye[level, :, :]
        count += 1
    max_val = np.max(layers)
    eye_norm = np.min(np.divide(layers, max_val), 0)
    resized_img = resize(eye_norm, (512, 512), anti_aliasing=True)
    plt.imshow(resized_img, cmap='gray')
    plt.show()


def single_slice(eye, level, orientation):
    if orientation == "depth":
        eye_slice = eye[level, :, :]
    elif orientation == "top":
        eye_slice = eye[:, level, :]
    elif orientation == "side":
        eye_slice = eye[:, :, level]
    else:
        return ValueError('InvalidOrientation')
    plt.imshow(eye_slice, cmap='gray')
    plt.show()


def load_data_set(src_dir, num_images):
    eye = np.ndarray(shape=(512, 512, num_images), dtype=float)
    for i in range(num_images):
        eye[:, :, i] = imread(join(src_dir, str(i + 1) + '.png'))
    return eye


if __name__ == '__main__':
    src = str(input("Enter the absolute path to the (OMAG) images you wish to process ... : "))
    n = len(listdir(src))
    if n == 0:
        raise ValueError('FoundZeroImages')
    print('loading {} images from `{}` ...'.format(n, src))
    eye = load_data_set(src, n)
    print("loading complete")
    multi_slice_sum(eye, 60, 120)
    multi_slice_min_norm(eye, 60, 120)
    fly_through(eye, range(60, 120, 10), 2, 3)
