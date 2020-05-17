# GPU device
GPU = '/device:GPU:0'

# Used by generator
OUTPUT_CHANNELS = 1
LAMBDA = 100

# Used in utils
IMAGE_DIM = 512

# Shuffle buffer size (>= dataset size for perfect shuffling)
BUFFER_SIZE = 1300
PIXEL_DEPTH = 256

# Directory names for B-Scans and OMAGs
BSCAN_DIRNAME = "xzIntensity"
OMAG_DIRNAME = "OMAG Bscans"
