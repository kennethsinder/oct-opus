import argparse
import datetime
import os
import time

import tensorflow as tf

from cgan.dataset import Dataset
from cgan.model_state import ModelState
from cgan.parameters import GPU, K_FOLDS_COUNT
from cgan.train import train_epoch
from cgan.utils import generate_inferred_images, generate_cross_section_comparison

# https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=[
                        'train', 'predict'], help='Specify the mode in which to run the program')
    parser.add_argument('-s', '--starting-epoch', type=int,
                        help='Specify the initial epoch number', default=1)
    parser.add_argument('-e', '--ending-epoch', type=int,
                        help='Specify the final epoch number', default=10)
    parser.add_argument('-d', '--datadir',
                        help='Specify the root directory to look for data')
    parser.add_argument('-c', '--ckptdir',
                        help='Optionally specify the location of the '
                             'checkpoints for prediction or to start off '
                             'the training', default=None)
    parser.add_argument('-k', '--k-folds', action='store_true',
                        help='Whether to use k-folds cross-validation '
                             'instead of treating all of the input '
                             'data as training data')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.k_folds:
        K_FOLDS_COUNT = 1

    # dataset
    assert args.datadir is not None
    ds = Dataset(root_data_path=args.datadir, num_folds=5)

    # main directory used to store output
    EXP_DIR = "experiment-{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    os.makedirs(EXP_DIR, exist_ok=False)
    with open(os.path.join(EXP_DIR, 'README.md'), 'w') as readme_file:
        # Create a mostly blank README file to encourage good
        # documentation of the purpose of each experiment.
        readme_file.write('# {}\n\n'.format(EXP_DIR))

    # TensorBoard
    TBD_WRITER = tf.summary.create_file_writer(os.path.join(EXP_DIR, "logs"))

    if args.mode == 'train':
        # First check there's a GPU available
        device_name = tf.test.gpu_device_name()
        if device_name != GPU:
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

        ckpt_dir = os.path.join(EXP_DIR, 'training_checkpoints')
        model_state = ModelState(exp_dir=EXP_DIR,
                                 ckpt_dir=ckpt_dir,
                                 dataset=ds)
        model_state.is_training_mode = True
        if args.ckptdir is not None:
            # Allow training to start off from an existing checkpoint,
            # say, from a different experiment or elsewhere if the
            # `ckptdir` command-line argument is supplied.
            model_state.restore_from_checkpoint(args.ckptdir)
        num_epochs = args.ending_epoch - args.starting_epoch + 1

        # go through each of K=5 folds, goes from 0 to 4 inclusive
        for fold_num in range(K_FOLDS_COUNT):
            print('----- Starting fold number {} -----'.format(fold_num))
            model_state.reset_weights()
            model_state.get_datasets(fold_num)

            # main epoch loop
            for epoch_num in range(args.starting_epoch, args.ending_epoch + 1):
                print('----- Starting epoch number {} -----'.format(epoch_num))
                start = time.time()
                train_epoch(TBD_WRITER, model_state.train_dataset,
                            model_state, epoch_num + fold_num * num_epochs)
                model_state.save_checkpoint()
                print('Time taken for epoch {} is {} sec\n'.format(
                    epoch_num, time.time() - start))

                # cross-section image logging
                for inp, tar in model_state.test_dataset.take(1):
                    generate_cross_section_comparison(EXP_DIR=EXP_DIR,
                                                      model=model_state.generator,
                                                      test_input=inp,
                                                      tar=tar,
                                                      epoch_num=epoch_num + fold_num * num_epochs)

            # Create predicted cross-section and enface images at the end of every fold.
            generate_inferred_images(EXP_DIR, model_state)
            print('Generated inferred images for fold {}'.format(fold_num))

        model_state.cleanup()   # Delete .h5 files for scrambled-weight models

    else:   # Testing Mode (i.e. Just generate predicted images, no training)

        # load from latest checkpoint and load data for just 1 of 5 folds
        assert args.ckptdir is not None
        model_state = ModelState(
            exp_dir=EXP_DIR, ckpt_dir=args.ckptdir, dataset=ds)
        model_state.is_training_mode = False
        model_state.restore_from_checkpoint()

        # generate results based on prediction
        generate_inferred_images(EXP_DIR, model_state)
