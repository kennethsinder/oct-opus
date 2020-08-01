import glob
import io
import os
import re
import tempfile
import shutil

from os.path import join
from random import randint
from typing import Iterable

import tensorflow as tf
from PIL import Image, ImageEnhance

from cgan.parameters import BUFFER_SIZE, IMAGE_DIM, PIXEL_DEPTH, BSCAN_DIRNAME, OMAG_DIRNAME, LAYER_BATCH
from cgan.random import resize, random_jitter, random_noise
from enface.enface import gen_single_enface


def get_dataset(root_data_path: str,
                dataset_iterable: Iterable) -> tf.data.Dataset:
    image_files = []
    dataset_names = []
    for dataset_name in dataset_iterable:
        matching_files = glob.glob(
            join(root_data_path, dataset_name, BSCAN_DIRNAME, '[0-9]*.png')
        )
        image_files.extend(matching_files)
        dataset_names.extend([dataset_name for _ in matching_files])

    dataset_names = tf.data.Dataset.from_tensor_slices(dataset_names)
    images = tf.data.Dataset.from_generator(
        lambda: map(get_images, image_files),
        output_types=(tf.float32, tf.float32),
    )

    # silently drop data that causes errors
    # (e.g. corresponding OMAG file doesn't exist)
    images = images.apply(tf.data.experimental.ignore_errors())

    dataset = tf.data.Dataset.zip((dataset_names, images))
    return dataset.shuffle(BUFFER_SIZE).batch(1)


def load_image(file_name, contrast_factor=1.0, sharpness_factor=1.0):
    """
    Decodes a grayscale PNG, returns a 2D tensor.
    """
    original_image = Image.open(file_name)

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


def get_num_acquisitions(data_folder_path: str) -> int:
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
    """ (int, int) -> int
    Takes in a 1-indexed B-scan ID and outputs a 1-indexed OMAG ID for
    the OMAG that would be paired with the B-scan in a dataset with a
    `num_acquisitions` B-scan:OMAG ratio.
    >>> print(bscan_num_to_omag_num(5, 4))
    2
    """
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
    path_components = re.search(r'^(.*)xzIntensity/(\d+)\.png$', bscan_path)

    dir_path = path_components.group(1)
    bscan_num = int(path_components.group(2))

    num_acquisitions = get_num_acquisitions(dir_path)

    omag_num = bscan_num_to_omag_num(bscan_num, num_acquisitions)


    # Construct a list of LAYER_BATCH BScans, where the central BScan
    # is the one found at bscan_path, and the others are a successive step away from
    # the central one. For instance, if LAYER_BATCH = 5, then bscans will take the form
    # [
    # <scan corresponding to omag_num - 2>
    # <scan corresponding to omag_num - 1>
    # <scan at bscan_path>,
    # <scan corresponding to omag_num + 1>
    # <scan corresponding to omag_num + 2>
    # ]
    bscans = []

    def omag_num_to_bscan(n):
        """ Return a random one of the num_acquisitions BScans
        corresponding to the nth OMAG """
        bn = (n - 1)  # convert to 0-index
        bn = bn * num_acquisitions  # convert to 0-indexed bscan num
        bn = bn + 1  # convert to 1-index

        # add a random offset corresponding to same OMAG
        return bn + randint(0, num_acquisitions - 1)

    for i in range(LAYER_BATCH):
        # If LAYER_BATCH = 3, then offset will range [-1, 0, 1],
        # if LAYER_BATCH = 5, then [-2, -1, 0, 1, 2], etc.
        offset = (i - LAYER_BATCH//2)

        if offset == 0:  # central BScan
            curr_bscan = bscan_num
        else:
            curr_omag = omag_num + offset
            curr_bscan = omag_num_to_bscan(curr_omag)

        curr_bscan_path = join(dir_path, 'xzIntensity', f'{curr_bscan}.png')

        try:
            bscans.append(load_image(
                curr_bscan_path, contrast_factor=1.85)[:,:,0])
        except FileNotFoundError:
            # We allow missing adjacent BScans; default value is instead all
            # zeros.
            if offset == 0:
                raise
            bscans.append(tf.zeros((IMAGE_DIM, IMAGE_DIM), dtype=float))

    bscans = [tf.cast(b, tf.float32) for b in bscans]
    bscan_img = tf.stack(bscans, axis=2)
    bscan_img = tf.cast(bscan_img, tf.float32)
    bscan_img = (bscan_img / ((PIXEL_DEPTH - 1) / 2.0)) - 1

    omag_img = load_image(
        join(dir_path, 'OMAG Bscans', '{}.png'.format(omag_num)),
        contrast_factor=1.85
    )
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


def generate_inferred_enface(dataset_dir: str,
                             output_dir: str,
                             generator: tf.keras.Model) -> None:
    """
    Generate predicted enfaces for the single dataset in dataset_dir.
    """

    """ Stage 1: Generate sequence of inferred OMAG-like images. """
    num_acquisitions = get_num_acquisitions(dataset_dir)
    dataset_name = os.path.basename(dataset_dir)
    bscans_list = glob.glob(join(dataset_dir, BSCAN_DIRNAME, '[0-9]*.png'))
    print(f"Found {len(bscans_list)} scans in {dataset_name}")

    tmpdir = tempfile.mkdtemp()

    # loop over each scan and generate corresponding predicted image
    for bscan_file_path in bscans_list:
        # Get number before '.png'
        bscan_id = int(re.search(r'(\d+)\.png', bscan_file_path).group(1))

        # We only want 1 out of every group of `num_acquisitions` B-scans to
        # gather a prediction from. In other words, if we have the OMAG ground
        # truth folder, we'll want the number of predicted OMAG-like images to
        # be the same as the number of real OMAGs.
        if bscan_id % num_acquisitions != 0:
            continue

        # Obtain a prediction of the image identified by filename `fn`.
        bscan_img, _ = get_images(bscan_file_path,
                                  use_random_jitter=False,
                                  use_random_noise=False)
        gen_output = generator(bscan_img[tf.newaxis, ...], training=False)
        predicted_img = gen_output[0, :, :, LAYER_BATCH//2, tf.newaxis]

        img_to_save = tf.image.encode_png(
            tf.dtypes.cast((predicted_img * 0.5 + 0.5) * (PIXEL_DEPTH - 1),
                           tf.uint8))

        # Save the prediction to disk under a sub-directory.
        omag_id = bscan_num_to_omag_num(bscan_id, num_acquisitions)
        tf.io.write_file(os.path.join(tmpdir, f"{omag_id}.png"),
                         img_to_save)

    """ Stage 2: Collect predicted OMAGs to make an en-face vascular map. """
    gen_single_enface(dataset_dir=tmpdir,
                      output_dir=output_dir,
                      output_prefix=dataset_name)

    shutil.rmtree(tmpdir)
