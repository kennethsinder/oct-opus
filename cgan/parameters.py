# GPU device
GPU = '/device:GPU:0'

# Used by generator
LAMBDA = 100
LAYER_BATCH = 1  # must be an odd number

# Used in utils, generator, discriminator
## The model's layers are chosen for 512x512 inputs, but it
## should work, if possibly suboptimally, with any input size.
IMAGE_DIM = 512

# Shuffle buffer size (>= dataset size for perfect shuffling)
BUFFER_SIZE = 1300
PIXEL_DEPTH = 256

# Directory names for B-Scans and OMAGs
BSCAN_DIRNAME = "xzIntensity"
OMAG_DIRNAME = "OMAG Bscans"
