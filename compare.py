from skimage.measure import compare_psnr, compare_mse, compare_nrmse, compare_ssim
import numpy as np
from PIL import Image


# TODO: peak absolute error
def pae(image_a, image_b):
    pass


# TODO: mean absolute error
def mae(image_a, image_b):
    pass


def compare_all(image_a, image_b):
    # peak signal to noise ratio
    psnr_score = compare_psnr(image_a, image_b)
    print("PSNR: {}".format(str(psnr_score)), end="\n\n")

    # mean squared error
    mse_score = compare_mse(image_a, image_b)
    print("MSE: {}".format(str(mse_score)))
    print("Range [0, +INF) where 0 is identical", end="\n\n")

    # normalized root mean squared error
    nrmse_score = compare_nrmse(image_a, image_b, norm_type='Euclidean')
    print("NRMSE: {}".format(str(nrmse_score)), end="\n\n")


    # structural similarity measure
    ssim_score = compare_ssim(image_a, image_b, full=False)
    print("SSIM: {}".format(str(ssim_score)))
    print("Range [-1, +1] where +1 is identical")


if __name__ == '__main__':
    image_a_path = str(input("Enter the absolute path of the 1st image : "))
    image_b_path = str(input("Enter the absolute path of the 2nd image : "))

    image_a_obj = Image.open(image_a_path)
    image_b_obj = Image.open(image_b_path)

    assert image_a_obj.size == image_b_obj.size
    print("Both images are " + str(image_a_obj.size))

    image_a = np.asarray(image_a_obj)
    image_b = np.asarray(image_b_obj)
    compare_all(image_a, image_b)
