import glob
import io
import re
from os import makedirs, listdir
from os.path import join, basename, normpath
from random import randint

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageEnhance

from cgan.parameters import BUFFER_SIZE, IMAGE_DIM, PIXEL_DEPTH
from cgan.random import resize, random_jitter, random_noise
from enface.enface import gen_single_enface


def get_dataset(data_dir: str, dataset_list):
    image_files = []
    for eye_path in [join(data_dir, eye_folder) for eye_folder in dataset_list]:
        image_files.extend(glob.glob(join(eye_path, 'xzIntensity', '[0-9]*.png')))

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


def generate_inferred_images(EXP_DIR, model_state):
    """
    Generate predicted images for all datasets listed under the `model_state.all_data_path` path.
    Also generates the corresponding enface for each of the new datasets.
    Images are saved under the `EXP_DIR`/<dataset_name> directory.
    """

    # lists the datasets found
    datasets_base_path = model_state.all_data_path
    datasets_list = listdir(datasets_base_path)
    print("Found {} datasets under {}".format(len(datasets_list), datasets_base_path))

    # loop over datasets
    for dataset_name in datasets_list:
        """ Stage 1: Generate sequence of inferred OMAG-like cross-section images. """
        bscans_list = glob.glob(join(datasets_base_path, dataset_name, 'xzIntensity', '[0-9]*.png'))
        print("Found {} scans belonging to {} dataset".format(len(bscans_list), dataset_name))
        makedirs(join(EXP_DIR, dataset_name), exist_ok=True)

        # loop over each scan and generate corresponding predicted image
        for bscan_file_path in bscans_list:
            # Get number before '.png'
            bscan_id = int(re.search(r'(\d+)\.png', bscan_file_path).group(1))

            # Obtain a prediction of the image identified by filename `fn`.
            bscan_img = load_image(bscan_file_path, angle=0, contrast_factor=1.85)
            bscan_img = (tf.cast(bscan_img, tf.float32) / ((PIXEL_DEPTH - 1) / 2.0)) - 1
            prediction = model_state.generator(bscan_img[tf.newaxis, ...], training=True)

            # Encode the prediction as PNG image data.
            predicted_img = prediction[0]
            img_to_save = tf.image.encode_png(tf.dtypes.cast((predicted_img * 0.5 + 0.5) * (PIXEL_DEPTH - 1), tf.uint8))

            # Save the prediction to disk under a sub-directory.
            tf.io.write_file('./{}/{}/{}.png'.format(EXP_DIR, dataset_name, bscan_id), img_to_save)

        """ Stage 2: Collect those predicted OMAGs to make an en-face vascular map. """
        path_to_predicted = join(EXP_DIR, dataset_name)
        gen_single_enface(dataset_dir=path_to_predicted)


def generate_cross_section_comparison(model, test_input, tar, epoch_num):
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

    figure_name = 'comparison_epoch_{}.png'.format(epoch_num)
    plt.savefig(figure_name)
    # TODO: replace with tensorboard
    # EXPERIMENT.log_figure(figure_name=figure_name)
    # EXPERIMENT.log_asset(file_data=figure_name, step=epoch_num)
    plt.clf()
