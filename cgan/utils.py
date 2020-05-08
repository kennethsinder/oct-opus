import glob
import io
import re
from os import makedirs
from os.path import join
from random import randint
from typing import Set

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageEnhance

from cgan.parameters import BUFFER_SIZE, IMAGE_DIM, PIXEL_DEPTH, BSCAN_DIRNAME, OMAG_DIRNAME
from cgan.random import resize, random_jitter, random_noise
from enface.enface import gen_single_enface


def get_dataset(root_data_path: str, dataset_iterable: Set):
    image_files = []
    for dataset_name in dataset_iterable:
        image_files.extend(glob.glob(join(root_data_path, dataset_name, BSCAN_DIRNAME, '[0-9]*.png')))

    if not image_files:
        raise Exception('Check src/parameters.py, no B-scan images were found.')
    dataset = tf.data.Dataset.from_generator(
        lambda: map(get_images, image_files),
        output_types=(tf.float32, tf.float32),
    )
    # silently drop data that causes errors (e.g. corresponding OMAG file doesn't exist)
    return dataset.apply(tf.data.experimental.ignore_errors()).shuffle(BUFFER_SIZE).batch(1)


def load_image(file_name, angle=0, contrast_factor=1.0, sharpness_factor=1.0):
    """
    Decodes a grayscale PNG, returns a 2D tensor.
    """
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
    if not omag_paths:
        # If we have no reference (ground truth) OMAGs available for this dataset,
        # that means it's unsuitable for training but may be suitable for generating
        # predictions from (e.g. a set of OCT images for a new eye for which OCTA/OMAG
        # was not used). In this case, let's assume that every B-scan is relevant,
        # returning 1 acquisition per spot. This also makes sure we don't divide
        # by zero below :)
        return 1
    return int(round(len(bscan_paths) / float(len(omag_paths))))


def bscan_num_to_omag_num(bscan_num: int, num_acquisitions: int) -> int:
    return ((bscan_num - 1) // num_acquisitions) + 1


def get_images(bscan_path, use_random_jitter=True, use_random_noise=False):
    """
    Returns a pair of tensors containing the given B-scan and its corresponding OMAG.
    Parameter `bscan_path` should be in directory 'xzIntensity' and its parent directory should contain 'OMAG Bscans'.
    Scan files should be named <num>.png (no leading 0s), with a 1-to-1 ratio of B-scans to OMAGs.
    """

    # random jitter angle
    angle = 0
    if use_random_jitter and tf.random.uniform(()) > 0.8:
        angle = randint(0, 45)

    # bscan image
    bscan_img = load_image(bscan_path, angle, contrast_factor=1.85)
    bscan_img = tf.cast(bscan_img, tf.float32)
    bscan_img = (bscan_img / ((PIXEL_DEPTH - 1) / 2.0)) - 1

    # path and image number
    path_components = re.search(r'^(.*)xzIntensity/(\d+)\.png$', bscan_path)
    dir_path = path_components.group(1)
    bscan_num = int(path_components.group(2))
    omag_num = bscan_num_to_omag_num(bscan_num, get_num_acquisitions(dir_path))

    # omag image
    omag_img = load_image(join(dir_path, OMAG_DIRNAME, '{}.png'.format(omag_num)), angle, contrast_factor=1.85)
    omag_img = tf.cast(omag_img, tf.float32)
    omag_img = (omag_img / ((PIXEL_DEPTH - 1) / 2.0)) - 1

    # random jitter
    if use_random_jitter:
        bscan_img, omag_img = random_jitter(bscan_img, omag_img)
    else:
        bscan_img, omag_img = resize(bscan_img, omag_img, IMAGE_DIM, IMAGE_DIM)

    # random noise
    if use_random_noise:
        # don't add noise to the omag image
        bscan_img = random_noise(bscan_img)
    return bscan_img, omag_img


def generate_inferred_images(EXP_DIR, model_state):
    """
    Generate predicted images for all datasets listed under the `model_state.all_data_path` path
    in test/predict mode, or for all *testing* datasets only when we're
    in training mode.
    Also generates the corresponding enface for each of the sets of predicted cross-sections
    that are generated.
    Images are saved under the `EXP_DIR`/<dataset_name> directory.
    """

    # determine which datasets we consider "testing" based on program mode
    if model_state.is_training_mode:
        dataset_names = model_state.test_folder_names
    else:
        dataset_names = model_state.DATASET.get_all_datasets()

    # loop over datasets
    for dataset_name in dataset_names:
        """ Stage 1: Generate sequence of inferred OMAG-like cross-section images. """

        dataset_path = join(model_state.DATASET.root_data_path, dataset_name)
        num_acquisitions = get_num_acquisitions(dataset_path)
        bscans_list = glob.glob(join(dataset_path, BSCAN_DIRNAME, '[0-9]*.png'))
        print("Found {} scans belonging to {} dataset".format(len(bscans_list), dataset_name))
        makedirs(join(EXP_DIR, dataset_name), exist_ok=True)

        # loop over each scan and generate corresponding predicted image
        for bscan_file_path in bscans_list:
            # Get number before '.png'
            bscan_id = int(re.search(r'(\d+)\.png', bscan_file_path).group(1))
            if bscan_id % num_acquisitions:
                # We only want 1 out of every group of `num_acquisitions` B-scans to gather
                # a prediction from. In other words, if we have the OMAG ground truth folder,
                # we'll want the number of predicted OMAG-like images to be the same as the
                # number of real OMAGs.
                continue

            # Obtain a prediction of the image identified by filename `fn`.
            bscan_img = load_image(bscan_file_path, angle=0, contrast_factor=1.85)
            bscan_img = (tf.cast(bscan_img, tf.float32) / ((PIXEL_DEPTH - 1) / 2.0)) - 1
            prediction = model_state.generator(bscan_img[tf.newaxis, ...], training=True)

            # Encode the prediction as PNG image data.
            predicted_img = prediction[0]
            img_to_save = tf.image.encode_png(tf.dtypes.cast((predicted_img * 0.5 + 0.5) * (PIXEL_DEPTH - 1), tf.uint8))

            # Save the prediction to disk under a sub-directory.
            omag_id = bscan_num_to_omag_num(bscan_id, num_acquisitions)
            tf.io.write_file('./{}/{}/{}.png'.format(EXP_DIR, dataset_name, omag_id), img_to_save)

        """ Stage 2: Collect those predicted OMAGs to make an en-face vascular map. """
        path_to_predicted = join(EXP_DIR, dataset_name)
        gen_single_enface(dataset_dir=path_to_predicted)


def generate_cross_section_comparison(EXP_DIR, model, test_input, tar, epoch_num):
    # the `training=True` is intentional here since
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

    figure_path = join(EXP_DIR, 'comparison_epoch_{}.png'.format(epoch_num))
    plt.savefig(figure_path)
    # TODO: replace with tensorboard
    # EXPERIMENT.log_figure(figure_name=figure_name)
    # EXPERIMENT.log_asset(file_data=figure_name, step=epoch_num)
    plt.clf()
