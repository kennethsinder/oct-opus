# September 2020 - cnn/parameters.py
#
# This code is an implemention of the convolutional neural network (CNN) approach
# of Dr. Aaron Y. Lee and their University of Washington team:
#
#     Lee, C.S., Tyring, A.J., Wu, Y., et al.
#     “Generating Retinal Flow Maps from Structural Optical Coherence Tomography with Artificial Intelligence,”
#     Scientific Reports 9, 5694 (2019).
#
# The paper can be found at: https://doi.org/10.1038/s41598-019-42042-y
# Credit goes to them, and we also thank them for helping us implement the model from their paper
# and particularly for their patience helping us reproduce the smaller details of their architecture.

GPU = '/device:GPU:0'

NUM_DATASETS = 66
NUM_IMAGES_PER_DATASET = 512

IMAGE_DIM = 512
PIXEL_DEPTH = 256

AUGMENT_NORMALIZE = 'normalize'
AUGMENT_CONTRAST = 'contrast'
AUGMENT_FULL = 'full_augment'

# format is <data-dir-name> -> <seed> -> <k-folds> -> <selected-fold> -> {mean, std}
STATS = {
    'single_poly_flattened' : {
        42 : {
            5 : [
                { # fold 0
                    'mean' : 0.9207166271894793,
                    'std' : 0.16064651068948066
                },
                { # fold 1
                    'mean' : 0.9201955425096048,
                    'std' : 0.16138837105477835
                },
                { # fold 2
                    'mean' : 0.9198806269935005,
                    'std' : 0.1613825064801529
                },
                { # fold 3
                    'mean' : 0.9189955722273949,
                    'std' : 0.1631408133232959
                },
                { # fold 4
                    'mean' : 0.9214165301668386,
                    'std' : 0.16016455591200518
                }
            ]
        }
    }
}
