# August 2020 - cgan/parameters.py
#
# This code is an adaptation of the approach from the 2017 pix2pix paper with some of our own additional small
# tweaks to the generator and discriminator model to suit our application.
#
#     Isola, P., Zhu, J., Zhou, T., Efros, A. A.,
#     "Image-to-Image Translation with Conditional Adversarial Networks,"
#     IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, 5967-5976 (2017).
#
# Significant additional credit also goes to TensorFlow's pix2pix tutorial page:
#
#     https://www.tensorflow.org/tutorials/generative/pix2pix
#
# We adapted a lot of code snippets from that tutorial, making changes as necessary to support our folder structure
# and use cases, as well as small model changes and refactoring the code over multiple files for maintainability.

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
