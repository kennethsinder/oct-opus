"""
Histogram equalize every image in every eye for every
predicted-epoch-# folder within the path p specified
as the only additional command line argument:
    ./post-normalizer.py <x>
"""

import cv2


def histogram(file_path):
    img = cv2.imread(file_path, 0)
    equ = cv2.equalizeHist(img)
    cv2.imwrite(file_path, equ)


if __name__ == '__main__':
    from glob import glob
    from os.path import join
    import sys

    for eye_folder in glob(join(sys.argv[1], '*')):
        for image_path in glob(join(eye_folder, '*.png')):
            histogram(image_path)
