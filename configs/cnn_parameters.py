from configs.parameters import IMAGE_DIM
from datasets.train_and_test import TESTING_DATASETS

DATASET_BLACKLIST = {
    '2015-10-21___512_2048_Horizontal_Images26',
    '2015-10-26___512_2048_Horizontal_Images43',
}

PREDICTION_IMAGES = {
    '2015-10-20___512_2048_Horizontal_Images15/xzIntensity/79.png',
    '2015-10-20___512_2048_Horizontal_Images83/xzIntensity/400.png',
    '2015-10-21___512_2048_Horizontal_Images31/xzIntensity/277.png'
}
assert {path[0:path.find('/')] for path in PREDICTION_IMAGES}.issubset(TESTING_DATASETS)

BATCH_SIZE = 50 #400

NUM_SLICES = 4

assert IMAGE_DIM % NUM_SLICES == 0
SLICE_WIDTH = IMAGE_DIM // NUM_SLICES
