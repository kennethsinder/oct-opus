import cv2
import numpy as np

import glob
import os
import os.path
import sys

for f in glob.glob(os.path.join(sys.argv[1], '*', '*', '*.png')):
    img = cv2.imread(f, 0)
    equ = cv2.equalizeHist(img)
    # Save histogram-equalized image in-place.
    cv2.imwrite(f, equ)
