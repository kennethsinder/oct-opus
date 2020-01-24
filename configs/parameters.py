from comet_ml import OfflineExperiment

# Comet Experiment
EXPERIMENT = OfflineExperiment(
    api_key="CnUAPYboS2Dbzv4j3qHkuxUev",
    project_name="oct-opus",
    offline_directory="./logs"
)

# GPU device
GPU = '/device:GPU:0'

# Training and testing data set locations
ALL_DATA_DIR = 'all_data_original'
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

# Rows we care about in each image
START_ROW = 50
END_ROW = 256

# Directory names for B-Scans and OMAGs
BSCAN_DIRNAME = "xzIntensity"
OMAG_DIRNAME = "OMAG Bscans"
