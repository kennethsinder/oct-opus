import glob
import os.path
import sys

import cv2

for f in glob.glob(os.path.join(sys.argv[1], '*.png')):
    img = cv2.imread(f, 0)
    equ = cv2.equalizeHist(img)
    print(os.path.basename(os.path.normpath(f)))
    cv2.imwrite(os.path.basename(os.path.normpath(f)), equ)
