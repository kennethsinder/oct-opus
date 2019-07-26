import os
import sys
import glob
import tensorflow as tf
from src.train import train, generator_optimizer, discriminator_optimizer
from src.utils import get_images, generate_inferred_images
from src.generator import generator
from src.discriminator import discriminator
from src.parameters import BUFFER_SIZE, TRAIN_DATA_DIR, TEST_DATA_DIR, OUTPUT_CHANNELS, EPOCHS

# Build a tf.data.Dataset of input B-scan and output OMAG images in the given directory.
def get_dataset(data_dir):
    dataset = tf.data.Dataset.from_generator(
        lambda: map(get_images, glob.glob(os.path.join(data_dir, '*', 'xzIntensity', '*.png'))),
        output_types=(tf.float32, tf.float32)
    )
    # silently drop data that causes errors (e.g. corresponding OMAG file doesn't exist)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(1)
    return dataset


if __name__ == '__main__':
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    # https://github.com/tensorflow/tensorflow/issues/1578#issuecomment-200544189
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    train_dataset = get_dataset(TRAIN_DATA_DIR)
    test_dataset = get_dataset(TEST_DATA_DIR)

    # get generator
    generator = generator(OUTPUT_CHANNELS)

    # get discriminator
    discriminator = discriminator()

    if 'train' in sys.argv:
        # train
        train(generator, discriminator, train_dataset, test_dataset, EPOCHS)

    if 'predict' in sys.argv:
        # load from checkpoint
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)
        checkpoint_dir = './training_checkpoints'
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        # generate results based on prediction
        generate_inferred_images(generator, TEST_DATA_DIR)
