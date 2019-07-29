import glob
import os
import os.path
import re
import io
import math
import tensorflow as tf
from src.parameters import BUFFER_SIZE
from src.train import discriminator_loss
from PIL import Image
from random import randint

IMAGE_DIM = 512
PIXEL_DEPTH = 256
NUM_ACQUISITIONS = 4


def get_dataset(data_dir):
    dataset = tf.data.Dataset.from_generator(
        lambda: map(get_images, glob.glob(os.path.join(
            data_dir, '*', 'xzIntensity', '*.png'))),
        output_types=(tf.float32, tf.float32)
    )
    # silently drop data that causes errors (e.g. corresponding OMAG file doesn't exist)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(1)
    return dataset


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMAGE_DIM, IMAGE_DIM, 1])
    return cropped_image[0], cropped_image[1]


@tf.function()
def random_jitter(input_image, real_image):
    if tf.random.uniform(()) > 0.5:
        input_image, real_image = resize(
            input_image, real_image, IMAGE_DIM + 30, IMAGE_DIM + 30)

        input_image, real_image = random_crop(input_image, real_image)
    else:
        input_image, real_image = resize(
            input_image, real_image, IMAGE_DIM, IMAGE_DIM)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


# Decodes a grayscale PNG, returns a 2D tensor.
def load_image(file_name, angle=0):
    if angle == 0:
        image = tf.io.read_file(file_name)
        image = tf.image.decode_png(image, channels=1)
        return image
    else:
        image = Image.open(file_name).rotate(angle)
        output = io.BytesIO()
        image.save(output, format='JPEG')
        return tf.image.decode_png(output.getvalue())


def bscan_num_to_omag_num(bscan_num):
    return ((bscan_num - 1) // NUM_ACQUISITIONS) + 1


def get_images(bscan_path, use_random_jitter=True):
    """
    Returns a pair of tensors containing the given B-scan and its
    corresponding OMAG. |bscan_path| should be in directory 'xzIntensity'
    and its parent directory should contain 'OMAG Bscans'. Scan files
    should be named <num>.png (no leading 0s), with a 4-to-1 ratio of
    B-scans to OMAGs.
    (OMAG Bscans/1.png corresponds to xzIntensity/{1,2,3,4}.png.)
    """
    angle = 0
    if use_random_jitter and tf.random.uniform(()) > 0.8:
        angle = randint(0, 45)
    bscan_img = load_image(bscan_path, angle)

    path_components = re.search(r'^(.*)xzIntensity/(\d+)\.png$', bscan_path)

    dir_path = path_components.group(1)
    bscan_num = int(path_components.group(2))

    omag_num = bscan_num_to_omag_num(bscan_num)

    omag_img = load_image(os.path.join(
        dir_path, 'OMAG Bscans', '{}.png'.format(omag_num)), angle)

    bscan_img = tf.cast(bscan_img, tf.float32)
    omag_img = tf.cast(omag_img, tf.float32)

    bscan_img = (bscan_img / ((PIXEL_DEPTH - 1) / 2.0)) - 1
    omag_img = (omag_img / ((PIXEL_DEPTH - 1) / 2.0)) - 1

    if use_random_jitter:
        bscan_img, omag_img = random_jitter(bscan_img, omag_img)
    else:
        bscan_img, omag_img = resize(
            bscan_img, omag_img, IMAGE_DIM, IMAGE_DIM)

    return bscan_img, omag_img


def get_images_no_jitter(bscan_path):
    return get_images(bscan_path, False)


def generate_inferred_images(model_state, test_data_dir):
    """
    Generate full sets of inferred cross-section PNGs,
    save them to /predicted/<dataset_name>_1.png -> /predicted/<dataset_name>_<N>.png
    where N is the number of input B-scans
    (i.e. 4 times the number of OMAGs we'd have for each test set).
    """
    for dataset_path in glob.glob(os.path.join(test_data_dir, '*')):
        dataset_name = dataset_path.split('/')[-1]
        for fn in glob.glob(os.path.join(dataset_path, 'xzIntensity', '*.png')):
            # Get number before '.png'
            i = int(re.search(r'(\d+)\.png', fn).group(1))
            if i % NUM_ACQUISITIONS:
                # We only want 1 out of every NUM_ACQUISITIONS B-scans to gather
                # a prediction from.
                continue
            # TODO: this is dumb, find a better way later (we have an issue
            # open that includes this).
            if not os.path.isfile(os.path.join(
                dataset_path, 'OMAG Bscans', '{}.png'.format(
                    bscan_num_to_omag_num(i)))):
                continue

            # Obtain a prediction of the image identified by filename `fn`.
            dataset = tf.data.Dataset.from_generator(
                lambda: map(get_images_no_jitter, [fn]),
                output_types=(tf.float32, tf.float32))
            dataset = dataset.batch(1)
            for inp, tar in dataset.take(1):
                pass
            prediction = model_state.generator(inp, training=True)

            # Compute the loss.
            disc_generated_output = model_state.discriminator(
                [inp, prediction], training=True)
            disc_real_output = model_state.discriminator(
                [inp, tar], training=True)
            disc_loss = discriminator_loss(
                disc_real_output, disc_generated_output)
            if i < 10:
                print('Discriminator loss: {}'.format(disc_loss))

            # Save the prediction to disk under a sub-directory.
            predicted_img = prediction[0]
            img_to_save = tf.image.encode_png(tf.dtypes.cast(
                (predicted_img * 0.5 + 0.5) * (PIXEL_DEPTH - 1), tf.uint8))
            os.makedirs('./predicted/{}'.format(dataset_name), exist_ok=True)
            write_op = tf.io.write_file('./predicted/{}/{}.png'.format(
                dataset_name, i // NUM_ACQUISITIONS + 1,
            ), img_to_save)
