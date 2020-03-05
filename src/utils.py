import glob
import io
import re
from os import makedirs
from os.path import join, basename, normpath, isfile
from random import randint

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageEnhance

from configs.parameters import BUFFER_SIZE, IMAGE_DIM, PIXEL_DEPTH, EXPERIMENT
from datasets.train_and_test import train_and_test_sets
from enface.enface import gen_enface_all_testing
from src.random import resize, random_jitter, random_noise
from src.train import discriminator_loss


def get_dataset(data_dir: str, dataset_list):
    image_files = glob.glob(join(data_dir, '*', 'xzIntensity', '*.png'))
    temp_image_files = []
    for image_path in image_files:
        dataset_name = last_path_component(join(image_path, '..', '..'))
        if dataset_name in dataset_list:
            temp_image_files.append(image_path)
    image_files = temp_image_files

    if not image_files:
        raise Exception('Check src/parameters.py, no B-scan images were found.')
    dataset = tf.data.Dataset.from_generator(
        lambda: map(get_images, image_files),
        output_types=(tf.float32, tf.float32),
    )
    # silently drop data that causes errors (e.g. corresponding OMAG file doesn't exist)
    return dataset.apply(tf.data.experimental.ignore_errors()).shuffle(BUFFER_SIZE).batch(1)


# Decodes a grayscale PNG, returns a 2D tensor.
def load_image(file_name, angle=0, contrast_factor=1.0, sharpness_factor=1.0):
    original_image = Image.open(file_name).rotate(angle)

    # contrast
    contrast_enhancer = ImageEnhance.Contrast(original_image)
    contrast_image = contrast_enhancer.enhance(contrast_factor)

    # sharpness
    sharpness_enhancer = ImageEnhance.Sharpness(contrast_image)
    sharpened_image = sharpness_enhancer.enhance(sharpness_factor)

    # write to buffer then tensor
    output = io.BytesIO()
    sharpened_image.save(output, format='png')
    return tf.image.decode_png(output.getvalue(), channels=1)


def get_num_acquisitions(data_folder_path):
    """ (str) -> int
    Auto-detect the number of acquisitions used for the data set in the
    folder identified by `data_folder_path`. Usually this will return
    the integer 1 or 4 (4 acquisitions is normal for OMAG).
    """
    bscan_paths = glob.glob(join(data_folder_path, 'xzIntensity', '*'))
    omag_paths = glob.glob(join(data_folder_path, 'OMAG Bscans', '*'))
    return int(round(len(bscan_paths) / float(len(omag_paths))))


def bscan_num_to_omag_num(bscan_num: int, num_acquisitions: int) -> int:
    return ((bscan_num - 1) // num_acquisitions) + 1


def get_images(bscan_path, use_random_jitter=True, use_random_noise=False):
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
    bscan_img = load_image(bscan_path, angle, contrast_factor=1.85)

    path_components = re.search(r'^(.*)xzIntensity/(\d+)\.png$', bscan_path)

    dir_path = path_components.group(1)
    bscan_num = int(path_components.group(2))

    omag_num = bscan_num_to_omag_num(bscan_num, get_num_acquisitions(dir_path))

    omag_img = load_image(join(dir_path, 'OMAG Bscans', '{}.png'.format(omag_num)), angle, contrast_factor=1.85)

    bscan_img = tf.cast(bscan_img, tf.float32)
    omag_img = tf.cast(omag_img, tf.float32)

    bscan_img = (bscan_img / ((PIXEL_DEPTH - 1) / 2.0)) - 1
    omag_img = (omag_img / ((PIXEL_DEPTH - 1) / 2.0)) - 1

    if use_random_jitter:
        bscan_img, omag_img = random_jitter(bscan_img, omag_img)
    else:
        bscan_img, omag_img = resize(bscan_img, omag_img, IMAGE_DIM, IMAGE_DIM)

    if use_random_noise:
        # don't add noise to the omag image
        bscan_img = random_noise(bscan_img)

    return bscan_img, omag_img


def get_images_no_jitter(bscan_path):
    return get_images(bscan_path, False)


def last_path_component(p: str) -> str:
    return basename(normpath(p))


def generate_inferred_images(model_state, epoch_num, fold_num=0):
    """
    Generate full sets of inferred cross-section PNGs,
    save them to /predicted/<dataset_name>_1.png -> /predicted/<dataset_name>_<N>.png
    where N is the number of input B-scans
    (i.e. 4 times the number of OMAGs we'd have for each test set).
    """
    test_folders = train_and_test_sets(fold_num)[1]
    predicted_dir = './predicted-epoch-{}/'.format(epoch_num)
    for dataset_path in glob.glob(join(model_state.all_data_path, '*')):
        dataset_name = last_path_component(dataset_path)
        if dataset_name not in test_folders:
            # Keep iterating if this data folder isn't configured to be in the
            # test set.
            continue
        for fn in glob.glob(join(dataset_path, 'xzIntensity', '*.png')):
            # Get number before '.png'
            i = int(re.search(r'(\d+)\.png', fn).group(1))
            num_acquisitions = get_num_acquisitions(dataset_path)
            if i % num_acquisitions:
                # We only want 1 out of every num_acquisitions B-scans to gather
                # a prediction from.
                continue
            # TODO: this is dumb, find a better way later (we have an issue open that includes this).
            if not isfile(join(dataset_path, 'OMAG Bscans', '{}.png'.format(
                    bscan_num_to_omag_num(i, num_acquisitions)))):
                continue

            # Obtain a prediction of the image identified by filename `fn`.
            dataset = tf.data.Dataset.from_generator(
                lambda: map(get_images_no_jitter, [fn]),
                output_types=(tf.float32, tf.float32)
            )
            dataset = dataset.batch(1)
            for inp, tar in dataset.take(1):
                pass
            prediction = model_state.generator(inp, training=True)

            # Compute the loss.
            disc_generated_output = model_state.discriminator([inp, prediction], training=True)
            disc_real_output = model_state.discriminator([inp, tar], training=True)

            # Save the prediction to disk under a sub-directory.
            predicted_img = prediction[0]
            img_to_save = tf.image.encode_png(tf.dtypes.cast((predicted_img * 0.5 + 0.5) * (PIXEL_DEPTH - 1), tf.uint8))

            makedirs(join(predicted_dir, dataset_name), exist_ok=True)
            tf.io.write_file('./{}/{}/{}.png'.format(predicted_dir, dataset_name, i // num_acquisitions + 1),
                             img_to_save)

    gen_enface_all_testing(predicted_dir, epoch_num, train_and_test_sets(fold_num))


def generate_cross_section_comparison(model, test_input, tar, epoch_num):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(tf.squeeze(display_list[i]) * 0.5 + 0.5)
        plt.axis('off')

    figure_name = 'comparison_epoch_{}.png'.format(epoch_num)
    plt.savefig(figure_name)
    EXPERIMENT.log_figure(figure_name=figure_name)
    EXPERIMENT.log_asset(file_data=figure_name, step=epoch_num)
    plt.clf()
