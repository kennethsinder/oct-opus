import glob
from os import makedirs
from os.path import basename, join, splitext, isfile

import tensorflow as tf

import cnn.utils as utils
import cnn.image as image

from enface.enface import gen_single_enface
from enface.image_io import MULTI_SLICE_SUM, MULTI_SLICE_MAX_NORM

def generate_enface(model, data_dir, verbose=False):
    if data_dir[-1] == '/':
        data_name = basename(data_dir[0:-1])
    else:
        data_name = basename(data_dir)
    enface_dir = join(
        model.enfaces_dir,
        'epoch_{}'.format(model.epoch.numpy()),
        data_name,
    )

    num_acquisitions = utils.get_num_acquisitions(data_dir)
    makedirs(enface_dir, exist_ok=True)

    # generate each cross section
    bscan_paths = glob.glob(join(data_dir, 'xzIntensity', '*.png'))
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

        dataset, num_batches = utils.load_dataset([bscan_path], 1, shuffle=False)
        img = model.predict(dataset, num_batches)
        image.save(img, join(enface_dir, '{}.png'.format(omag_num)))

    # using the cross sections, generate the enfaces
    gen_single_enface(enface_dir)

    # log enfaces to tensorboard
    with model.writer.as_default():
        # log max norm enface
        img = image.load(join(enface_dir, MULTI_SLICE_MAX_NORM))
        w, h, c = img.shape
        img = tf.reshape(img, [1, w, h, c])
        tf.summary.image('MULTI_SLICE_MAX_NORM', img, step=model.epoch.numpy())
        # log sum enface
        img = image.load(join(enface_dir, MULTI_SLICE_SUM))
        w, h, c = img.shape
        img = tf.reshape(img, [1, w, h, c])
        tf.summary.image('MULTI_SLICE_SUM', img, step=model.epoch.numpy())
