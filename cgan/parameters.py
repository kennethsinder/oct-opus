
# GPU device
GPU = '/device:GPU:0'

# Cross-validation modes
USE_K_FOLDS = True

# Training and testing data set locations
USE_FLATTENED = True
ALL_DATA_DIR = 'all_data_flattened' if USE_FLATTENED else 'all_data_original'
ENFACE_DATA_DIR = 'all_data_enface'

# Used by generator
OUTPUT_CHANNELS = 1
LAMBDA = 100

# Used in utils
IMAGE_DIM = 512

# Shuffle buffer size (>= dataset size for perfect shuffling)
BUFFER_SIZE = 1300
PIXEL_DEPTH = 256

# Used by Tensorboard
IMAGE_LOG_INTERVAL = 1000

# Directory names for B-Scans and OMAGs
BSCAN_DIRNAME = "xzIntensity"
OMAG_DIRNAME = "OMAG Bscans"
