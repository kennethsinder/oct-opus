import argparse
import glob
from datetime import datetime
from os import makedirs
from os.path import basename, isfile, join, splitext

import numpy as np
import tensorflow as tf
from PIL import Image

import cnn.utils as utils
from cnn.model import CNN
from cnn.parameters import (
    BATCH_SIZE,
    IMAGE_DIM,
    GPU,
    MULTI_SLICE_MAX_NORM,
    MULTI_SLICE_SUM,
    ROOT_ENFACE_DIR
)

from enface.enface import gen_single_enface


def generate_enface(model, data_dir):
    data_name = basename(data_dir)
    enface_dir = join(
        ROOT_ENFACE_DIR,
        basename(model.checkpoints_dir),
        str(model.epoch_num()),
        data_name,
    )
    num_acquisitions = utils.get_num_acquisitions(data_dir)
    makedirs(enface_dir, exist_ok=True)

    for bscan_path in glob.glob(join(data_dir, 'xzIntensity', '*.png')):
        bscan_num = int(splitext(basename(bscan_path))[0])
        if bscan_num % num_acquisitions:
            continue

        omag_num = utils.bscan_num_to_omag_num(bscan_num, num_acquisitions)

        omag_path = join(
            data_dir,
            'OMAG Bscans',
            '{}.png'.format(omag_num)
        )
        if not isfile(omag_path):
            continue

        model.predict(
            bscan_path,
            join(enface_dir, '{}.png'.format(omag_num))
        )

    gen_single_enface(enface_dir)

    #TODO: log images to tensorboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'enface'])
    parser.add_argument('hardware', choices=['cpu', 'gpu'])
    parser.add_argument('-d', '--data-dir', required=True, metavar='<path>')
    parser.add_argument('-s', '--split', type=float, default=0.8, metavar='<float>')
    parser.add_argument('-b', '--batch', type=int, default=BATCH_SIZE, metavar='<int>')
    parser.add_argument(
        '-c', '--checkpoints-dir',
        default=join('./checkpoints', datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')),
        metavar='<path>'
    )
    parser.add_argument('-e', '--num-epochs', type=int, default=1, metavar='<int>')
    parser.add_argument('-ef', '--enface-dir', required=True, metavar='<path>')
    args = parser.parse_args()

    if args.hardware == 'gpu':
        device_name = tf.test.gpu_device_name()
        if device_name != GPU:
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    model = CNN(args.data_dir, args.split, args.batch, args.checkpoints_dir)

    if args.mode == 'train':
        print('Saving checkpoints to {}'.format(model.checkpoints_dir))
        if args.num_epochs < 1:
            raise('Number of epochs must be at least one.')
        history = model.train(args.num_epochs)
    else:
        generate_enface(model, args.enface_dir)


if __name__ == '__main__':
    main()
