from datasets.train_and_test import TESTING_DATASETS

GPU = '/device:GPU:0'

IMAGE_DIM = 512
BUFFER_SIZE = 1300

DATASET_BLACKLIST = {
    '2015-10-21___512_2048_Horizontal_Images26',
    '2015-10-26___512_2048_Horizontal_Images43',
}

BATCH_SIZE = 50 #400
NUM_SLICES = 4

assert IMAGE_DIM % NUM_SLICES == 0
SLICE_WIDTH = IMAGE_DIM // NUM_SLICES

PREDICTIONS_BASE_DIR='./predictions'

START_ROW = 50
END_ROW = 256
MULTI_SLICE_MAX_NORM = "multi_slice_max_norm.png"
MULTI_SLICE_SUM = "multi_slice_sum.png"
