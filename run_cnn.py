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
    GPU,
    MULTI_SLICE_MAX_NORM,
    MULTI_SLICE_SUM,
    PREDICTIONS_BASE_DIR
)

from enface.image_io import ImageIO
from enface.slicer import Slicer

def generate_enfaces(model, data_dir):
    predictions_dir = join(
        PREDICTIONS_BASE_DIR,
        basename(model.checkpoints_dir),
        str(model.epoch_num())
    )

    for idx, dataset_name in enumerate(TESTING_DATASETS):
        print('({}/{}) Generating enface images for {}'.format(
                idx + 1,
                len(TESTING_DATASETS),
                dataset_name
            )
        )

        dataset_predictions_dir = join(predictions_dir, dataset_name)
        dataset_dir = join(data_dir, dataset_name)
        num_acquisitions = utils.get_num_acquisitions(dataset_dir)

        makedirs(dataset_predictions_dir, exist_ok=True)

        for bscan_path in glob.glob(join(dataset_dir, 'xzIntensity', '*.png')):
            bscan_num = int(splitext(basename(bscan_path))[0])
            if bscan_num % num_acquisitions:
                continue

            omag_num = utils.bscan_num_to_omag_num(bscan_num, num_acquisitions)

            omag_path = join(
                dataset_dir,
                'OMAG Bscans',
                '{}.png'.format(omag_num)
            )
            if not isfile(omag_path):
                continue

            model.predict(
                bscan_path,
                join(dataset_predictions_dir, '{}.png'.format(omag_num))
            )

        image_io = ImageIO()
        slicer = Slicer()
        eye = image_io.load_single_eye(dataset_predictions_dir)

        multi_slice_max_norm = slicer.multi_slice_max_norm(
            eye=eye,
            lower=START_ROW,
            upper=END_ROW
        )
        multi_slice_sum = slicer.multi_slice_sum(
            eye=eye,
            lower=START_ROW,
            upper=END_ROW
        )

        image_io.save_enface_image(
            enface=multi_slice_sum,
            filepath=dataset_predictions_dir,
            filename=MULTI_SLICE_SUM
        )
        image_io.save_enface_image(
            enface=multi_slice_max_norm,
            filepath=dataset_predictions_dir,
            filename=MULTI_SLICE_MAX_NORM
        )

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
        return

    generate_enfaces(model, args.data_dir)


if __name__ == '__main__':
    main()
