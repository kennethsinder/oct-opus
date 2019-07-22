import os
import re
import tensorflow as tf


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image):
    IMAGE_DIM = 512
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMAGE_DIM, IMAGE_DIM, 1])
    return cropped_image[0], cropped_image[1]


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 572 x 572
    input_image, real_image = resize(input_image, real_image, 572, 572)

    # randomly cropping to 512 x 512
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


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

    bscan_img = (bscan_img / 255.5) - 1
    omag_img = (omag_img / 255.5) - 1

    bscan_img, omag_img = random_jitter(bscan_img, omag_img)

    return bscan_img, omag_img

def generate_inferred_images(generator, test_data_dir):
    # Generate a full set of inferred cross-section PNGs, save them to /predicted/1.png -> /predicted/<N>.png
    # where N is the number of input B-scans (so 4 times the number of OMAGs we'd have for this test set).
    for i, fn in enumerate(glob.glob(os.path.join(test_data_dir, '*', 'xzIntensity', '*.png'))):
        dataset = tf.data.Dataset.from_generator(
            lambda: map(get_images_no_jitter, [fn]),
            output_types=(tf.float32, tf.float32))
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.batch(1)
        for inp, _ in dataset.take(1):
            pass

        prediction = generator(inp, training=True)
        predicted_img = prediction[0]
        img_to_save = tf.image.encode_png(tf.dtypes.cast((predicted_img * 0.5 + 0.5) * 255, tf.uint8))
        write_op = tf.io.write_file('./predicted/{}.png'.format(
            re.search(r'(\d+)\.png', fn).group(1)
        ), img_to_save)
