# Set these to be the directories containing the data subdirectories
# (so xzIntensity and OMAG BScans are 2 levels down from this dir).
TRAIN_DATA_DIR = './train'
TEST_DATA_DIR = './test'

# Shuffle buffer size (>= dataset size for perfect shuffling)
BUFFER_SIZE = 400

# This should actually be much more but also that will blow up the RAM
EPOCHS = 100

# Used by generator
OUTPUT_CHANNELS = 1

