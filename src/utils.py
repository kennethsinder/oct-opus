import os
import re
import tensorflow as tf


# Decodes a grayscale PNG, returns a 2D tensor.
def load_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_png(image)
    return image


# Returns a pair of tensors containing the given B-scan and its
# corresponding OMAG. |bscan_path| should be in directory 'xzIntensity'
# and its parent directory should contain 'OMAG Bscans'. Scan files
# should be named <num>.png (no leading 0s), with a 4-to-1 ratio of
# B-scans to OMAGs.
# (OMAG Bscans/1.png corresponds to xzIntensity/{1,2,3,4}.png.)
def get_images(bscan_path):
    bscan_img = load_image(bscan_path)
    path_components = re.search(r'^(.*)xzIntensity/(\d+)\.png$', bscan_path)

    dir_path = path_components.group(1)
    bscan_num = int(path_components.group(2))

    omag_num = ((bscan_num - 1) // 4) + 1
    omag_img = load_image(os.path.join(dir_path, 'OMAG Bscans', '{}.png'.format(omag_num)))

    bscan_img = tf.cast(bscan_img, tf.float32)
    omag_img = tf.cast(omag_img, tf.float32)

    return bscan_img, omag_img
