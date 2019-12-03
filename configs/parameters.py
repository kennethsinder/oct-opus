# GPU device
GPU = '/device:GPU:0'

# Training and testing data set locations
ALL_DATA_DIR = '/private/fydp1/all_data_normalized'
ENFACE_DATA_DIR = '/private/fydp1/enface_data'

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

# Directory names for B-Scans and OMAGs
BSCAN_DIRNAME = "xzIntensity"
OMAG_DIRNAME = "OMAG Bscans"

# Whether we're doing training and testing with enfaces vs. cross-sections
IS_ENFACE_TRAINING = False
