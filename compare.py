from skimage.measure import compare_psnr, compare_mse, compare_nrmse, compare_ssim
import numpy as np
from PIL import Image
import sys


def fsim(image_a, image_b):
    pass


def pae(image_a, image_b):
    # note: this isn't actually particularly well defined or commonly used metric, but including it just in case
    return np.max(np.absolute((image_a.astype("float") - image_b.astype("float"))))


def mae(image_a, image_b):
    # see https://en.wikipedia.org/wiki/Mean_absolute_error
    error_sum = np.sum(np.absolute(
        (image_a.astype("float") - image_b.astype("float"))))
    return error_sum / (image_a.shape[0] * image_a.shape[1])


def compare_all(image_a, image_b):
    # peak signal to noise ratio (see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
    psnr_score = compare_psnr(image_a, image_b)
    print("PSNR: {}".format(str(psnr_score)), end=" dB\n\n")

    # mean squared error (see https://en.wikipedia.org/wiki/Mean_squared_error)
    mse_score = compare_mse(image_a, image_b)
    print("MSE: {}".format(str(mse_score)))
    print("Range [0, +INF) where 0 is identical", end="\n\n")

    # normalized root mean squared error (see https://en.wikipedia.org/wiki/Root-mean-square_deviation)
    nrmse_score = compare_nrmse(image_a, image_b, norm_type='Euclidean')
    print("NRMSE: {}".format(str(nrmse_score)), end="\n\n")

    # structural similarity measure (see https://en.wikipedia.org/wiki/Structural_similarity)
    ssim_score = compare_ssim(image_a, image_b, full=False)
    print("SSIM: {}".format(str(ssim_score)))
    print("Range [-1, +1] where +1 is identical", end="\n\n")

    pae_score = pae(image_a, image_b)
    print("PAE: {}".format(str(pae_score)))
    print("Range [0, +INF) where 0 is identical", end="\n\n")

    mae_score = mae(image_a, image_b)
    print("MAE: {}".format(str(mae_score)))
    print("Range [0, +INF) where 0 is identical", end="\n\n")


def main(image_a_path, image_b_path):
    image_a_obj = Image.open(image_a_path)
    image_b_obj = Image.open(image_b_path)
    assert image_a_obj.size == image_b_obj.size

    print("Both images are " + str(image_a_obj.size), end="\n\n")

    image_a = np.asarray(image_a_obj)
    image_b = np.asarray(image_b_obj)
    compare_all(image_a, image_b)


if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2])
    except:
        raise Exception(
            'Usage: {} <file 1 path> <file 2 path>'.format(sys.argv[0]))
