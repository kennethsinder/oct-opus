import sys
from enface.enface import gen_single_enface

"""
Generates enface images from cross-sectional scans (e.g. B-Scans or OMAGs).
Resulting enface images are saved in the directory the images were sourced from.
"""

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: python enface.py [path to images]")
        exit(1)
    gen_single_enface(sys.argv[1])
