GPU = '/device:GPU:0'

# Training and testing data set locations
ALL_DATA_DIR = '/private/fydp1/all_data'
ENFACE_DATA_DIR = '/private/fydp1/enface_data'

# This should actually be much more but also that will blow up the RAM
# CURRENTLY UNUSED! See https://github.com/kennethsinder/oct-opus/pull/35
EPOCHS = 50

# Used by generator
OUTPUT_CHANNELS = 1
LAMBDA = 100

# Used in utils
IMAGE_DIM = 512
# Shuffle buffer size (>= dataset size for perfect shuffling)
BUFFER_SIZE = 1300
PIXEL_DEPTH = 256

# Used by Tensorboard
IMAGE_LOG_INTERVAL = 100

# Rows we care about in each image
START_ROW = 50
END_ROW = 256

# Train/test split
DATA_CONFIG = {
    "train": [
        "2015-10-23___512_2048_Horizontal_Images9",
        "2015-10-23___512_2048_Horizontal_Images14",
        "2015-10-23___512_2048_Horizontal_Images32",
        "2015-10-27___512_2048_Horizontal_Images67",
    ],
    "test": [
        "2015-10-23___512_2048_Horizontal_Images64",
        "2015-10-23___512_2048_Horizontal_Images37",
        "2015-10-22___512_2048_Horizontal_Images41",
    ]
}
