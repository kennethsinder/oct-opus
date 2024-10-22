# September 2020 - cnn/enface.py
#
# This code is an implemention of the convolutional neural network (CNN) approach
# of Dr. Aaron Y. Lee and their University of Washington team:
#
#     Lee, C.S., Tyring, A.J., Wu, Y., et al.
#     “Generating Retinal Flow Maps from Structural Optical Coherence Tomography with Artificial Intelligence,”
#     Scientific Reports 9, 5694 (2019).
#
# The paper can be found at: https://doi.org/10.1038/s41598-019-42042-y
# Credit goes to them, and we also thank them for helping us implement the model from their paper
# and particularly for their patience helping us reproduce the smaller details of their architecture.

import glob
from os import makedirs
from os.path import basename, join, splitext, isfile

import tensorflow as tf

import cnn.utils as utils
import cnn.image as image
from cnn.parameters import AUGMENT_NORMALIZE

from enface.enface import gen_single_enface
from enface.image_io import MULTI_SLICE_SUM, MULTI_SLICE_MAX_NORM

def generate_enface(model, data_dir, normalize=False, verbose=False):
    if data_dir[-1] == '/':
        data_name = basename(data_dir[:-1])
    else:
        data_name = basename(data_dir)

    enface_dir = join(
        model.enfaces_dir,
        'epoch_{}'.format(model.epoch.numpy()),
        data_name
    )

    num_acquisitions = utils.get_num_acquisitions(data_dir)
    makedirs(enface_dir, exist_ok=True)

    # generate each cross section
    bscan_paths = glob.glob(join(data_dir, 'xzIntensity', '[0-9]*.png'))
    for idx, bscan_path in enumerate(bscan_paths):
        bscan_num = int(splitext(basename(bscan_path))[0])
        if bscan_num % num_acquisitions:
            if verbose:
                print('{}/{} - Skipping {}: generating for only every 1 of {} bscans'.format(
                    idx + 1, len(bscan_paths), bscan_path, num_acquisitions))
            continue

        omag_num = utils.bscan_num_to_omag_num(bscan_num, num_acquisitions)
        omag_path = join(data_dir, 'OMAG Bscans', '{}.png'.format(omag_num))
        if not isfile(omag_path):
            if verbose:
                print('{}/{} - Skipping {}: cannot find corresponding omag'.format(
                    idx + 1, len(bscan_paths), bscan_path))
            continue

        if verbose:
            print('{}/{} - Generating cross section for {}'.format(
                idx + 1, len(bscan_paths), bscan_path))

        if model.augment_level == AUGMENT_NORMALIZE:
            dataset, num_batches = utils.load_dataset(
                [bscan_path],
                batch_size=1,
                num_slices=model.slices,
                mean=model.mean,
                standard_deviation=model.std,
                shuffle=False
            )
        else:
            dataset, num_batches = utils.load_augmented_dataset(
                [bscan_path],
                batch_size=1,
                num_slices=model.slices,
                use_random_jitter=False,
                use_random_noise=False,
                shuffle=False
            )

        # predicted image has shape [C,H,W]
        img = model.predict(dataset, num_batches)
        image.save(
            img,
            path=join(enface_dir, '{}.png'.format(omag_num)),
            data_format='channels_first'
        )

    # using the cross sections, generate the enfaces
    gen_single_enface(enface_dir, normalize=normalize)

    # log enfaces to tensorboard
    with model.writer.as_default():
        # log max norm enface
        img = image.load(join(enface_dir, MULTI_SLICE_MAX_NORM))
        w, h, c = img.shape
        img = tf.reshape(img, [1, w, h, c])
        tf.summary.image('{}_MULTI_SLICE_MAX_NORM'.format(data_name), img, step=model.epoch.numpy())
        # log sum enface
        img = image.load(join(enface_dir, MULTI_SLICE_SUM))
        w, h, c = img.shape
        img = tf.reshape(img, [1, w, h, c])
        tf.summary.image('{}_MULTI_SLICE_SUM'.format(data_name), img, step=model.epoch.numpy())
