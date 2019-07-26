# Training and testing data set locations
TRAIN_DATA_DIR = '/private/fydp1/training-data'
TEST_DATA_DIR = '/private/fydp1/testing-data'

# Shuffle buffer size (>= dataset size for perfect shuffling)
BUFFER_SIZE = 400

# This should actually be much more but also that will blow up the RAM
# CURRENTLY UNUSED! See https://github.com/kennethsinder/oct-opus/pull/35
EPOCHS = 50

# Used by generator
OUTPUT_CHANNELS = 1
