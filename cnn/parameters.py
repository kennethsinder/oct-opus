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
