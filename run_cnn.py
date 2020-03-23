from configs.parameters import EXPERIMENT
assert EXPERIMENT.alive  # Needed to due import dependency issues

import argparse
from datetime import datetime
from os.path import join

import numpy as np
import tensorflow as tf
from PIL import Image

import cnn.utils as utils
from cnn.model import CNN
from configs.parameters import GPU
from configs.cnn_parameters import PREDICTION_IMAGES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    parser.add_argument('hardware', choices=['cpu', 'gpu'])
    parser.add_argument('-d', '--data-dir', required=True, metavar='<path>')
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

    model = CNN(args.data_dir, args.checkpoints_dir)

    if args.mode == 'train':
        print('Saving checkpoints to {}'.format(args.checkpoints_dir))
        for _ in range(args.num_epochs):
            print('----------------- Epoch {} -----------------'.format(model.epoch.numpy()))
            training_loss = model.train_one_epoch()
            model.save_checkpoint()

            print('Test')
            testing_loss = model.test()

            print('Saving Predictions')
            model.predict(PREDICTION_IMAGES)

            model.increment_epoch()
        return

    model.predict(PREDICTION_IMAGES)

if __name__ == '__main__':
    main()
