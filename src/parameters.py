GPU = '/device:GPU:0'

# Training and testing data set locations
TRAIN_DATA_DIR = '/private/fydp1/training-data'
TEST_DATA_DIR = '/private/fydp1/testing-data'

# This should actually be much more but also that will blow up the RAM
# CURRENTLY UNUSED! See https://github.com/kennethsinder/oct-opus/pull/35
EPOCHS = 50

# Used by generator
OUTPUT_CHANNELS = 1
LAMBDA = 100

# Used in utils
IMAGE_DIM = 512
BUFFER_SIZE = 400 # Shuffle buffer size (>= dataset size for perfect shuffling)
PIXEL_DEPTH = 256
NUM_ACQUISITIONS = 4

# Used by Tensorboard
SCALAR_LOG_INTERVAL = 10
IMAGE_LOG_INTERVAL = 10

# Rows we care about in each image
START_ROW = 50
END_ROW = 256
