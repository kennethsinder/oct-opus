import os
import sys
import glob
import tensorflow as tf
from src.train import train
from src.model_state import ModelState
from src.utils import generate_inferred_images
from src.parameters import BUFFER_SIZE, EPOCHS, TEST_DATA_DIR


if __name__ == '__main__':
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    model_state = ModelState()

    if 'restore' in sys.argv and 'predict' not in sys.argv:
        model_state.restore_from_checkpoint()

    if 'train' in sys.argv:
        # train
        train(model_state, int(sys.argv[-1]))

    if 'predict' in sys.argv:
        # load from checkpoint
        model_state.restore_from_checkpoint()

        # generate results based on prediction
        generate_inferred_images(generator, TEST_DATA_DIR)
