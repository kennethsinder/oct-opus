import os
import glob
import tensorflow as tf
from src.utils import get_images
from src.generator import generator
from src.discriminator import discriminator
from src.train import train


# Build a tf.data.Dataset of input B-scan and output OMAG images in the given directory.
def get_dataset(data_dir):
    dataset = tf.data.Dataset.from_generator(
        lambda: map(get_images, glob.glob(os.path.join(data_dir, '*', 'xzIntensity', '*.png'))),
        output_types=(tf.float32, tf.float32))
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

    # Set these to be the directories containing the data subdirectories
    # (so xzIntensity and OMAG BScans are 2 levels down from this dir).
    TRAIN_DATA_DIR = './train'
    TEST_DATA_DIR = './test'

    # Shuffle buffer size (>= dataset size for perfect shuffling)
    BUFFER_SIZE = 400

    # This should actually be much more but also that will blow up the RAM
    EPOCHS = 100

    train_dataset = get_dataset(TRAIN_DATA_DIR)
    test_dataset = get_dataset(TEST_DATA_DIR)

    # get generator
    OUTPUT_CHANNELS = 1
    generator = generator(OUTPUT_CHANNELS)
    # gen_output = generator(input_img[tf.newaxis, ...], training=False)

    # get discriminator
    discriminator = discriminator()
    # disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)

    # train
    train(generator, discriminator, train_dataset, test_dataset, EPOCHS)
