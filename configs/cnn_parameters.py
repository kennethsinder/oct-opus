from configs.parameters import IMAGE_DIM


BATCH_SIZE = 400

NUM_SLICES = 4

assert IMAGE_DIM % NUM_SLICES == 0
SLICE_WIDTH = IMAGE_DIM // NUM_SLICES
