# August 2020 - cgan.py
#
# This code is an adaptation of the approach from the 2017 pix2pix paper with some of our own additional small
# tweaks to the generator and discriminator model to suit our application.
#
#     Isola, P., Zhu, J., Zhou, T., Efros, A. A.,
#     "Image-to-Image Translation with Conditional Adversarial Networks,"
#     IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, 5967-5976 (2017).
#
# Significant additional credit also goes to TensorFlow's pix2pix tutorial page:
#
#     https://www.tensorflow.org/tutorials/generative/pix2pix
#
# We adapted a lot of code snippets from that tutorial, making changes as necessary to support our folder structure
# and use cases, as well as small model changes and refactoring the code over multiple files for maintainability.

import argparse
import datetime
import os
import time

import tensorflow as tf

from cgan.dataset import Dataset
from cgan.model_state import ModelState
from cgan.parameters import GPU
from cgan.train import train_epoch
from cgan.utils import generate_inferred_images, generate_cross_section_comparison

# https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'], help='Specify the mode in which to run the program')
    parser.add_argument('-e', '--num-epochs', type=int, help='Specify the number of epochs of training to run',
                        default=3)
    parser.add_argument('-d', '--datadir', help='Specify the root directory to look for data')
    parser.add_argument('-c', '--ckptdir', help='Optionally specify the location of the '
                                                'checkpoints for prediction or to start off '
                                                'the training', default=None)
    parser.add_argument('-k', '--k-folds', type=int, help='Specify the number of folds to divide the data '
                                                          'into as part of K-Folds Cross-Validation', default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # dataset
    assert args.datadir is not None
    ds = Dataset(root_data_path=args.datadir, num_folds=args.k_folds)

    # main directory used to store output
    EXP_DIR = "experiment-{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    os.makedirs(EXP_DIR, exist_ok=False)
    with open(os.path.join(EXP_DIR, 'README.md'), 'w') as readme_file:
        # Create a mostly blank README file to encourage good
        # documentation of the purpose of each experiment.
        readme_file.write('# {}\n\n'.format(EXP_DIR))

    if args.k_folds > 1:
        # If training with k-folds, save the dataset partitions configuration
        # to a JSON file so we know which predicted eye came from which fold.
        import json

        def serialize_sets(obj):
            if isinstance(obj, set):
                return list(obj)
        fold_config = {i: ds.get_train_and_test_by_fold_id(i) for i in range(args.k_folds)}
        with open(os.path.join(EXP_DIR, 'fold_config.json'), 'w') as outfile:
            json.dump(fold_config, outfile, default=serialize_sets)

    # TensorBoard
    TBD_WRITER = tf.summary.create_file_writer(os.path.join(EXP_DIR, "logs"))

    if args.mode == 'train':
        # First check there's a GPU available
        device_name = tf.test.gpu_device_name()
        if device_name != GPU:
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

        ckpt_dir = os.path.join(EXP_DIR, 'training_checkpoints')
        model_state = ModelState(is_training_mode=True,
                                 ckpt_dir=ckpt_dir,
                                 dataset=ds)
        if args.ckptdir is not None:
            # Allow training to start off from an existing checkpoint,
            # say, from a different experiment or elsewhere if the
            # `ckptdir` command-line argument is supplied.
            print('Restoring from checkpoint at {}'.format(args.ckptdir))
            model_state.restore_from_checkpoint(args.ckptdir)

        # go through each of K=5 folds, goes from 0 to 4 inclusive
        for fold_num in range(args.k_folds):
            print('----- Starting fold number {} -----'.format(fold_num))
            model_state.reset_weights()
            model_state.get_datasets(fold_num)

            # main epoch loop
            for epoch_num in range(1, args.num_epochs + 1):
                print('----- Starting epoch number {} -----'.format(epoch_num))
                start = time.time()
                train_epoch(TBD_WRITER, model_state.train_dataset, model_state, epoch_num, fold_num)
                model_state.save_checkpoint()
                print('Time taken for epoch {} is {} sec\n'.format(
                    epoch_num, time.time() - start))

                # cross-section image logging
                for inp, tar in model_state.test_dataset.take(1):
                    generate_cross_section_comparison(EXP_DIR=EXP_DIR,
                                                      TBD_WRITER=TBD_WRITER,
                                                      model=model_state.generator,
                                                      test_input=inp,
                                                      tar=tar,
                                                      epoch_num=epoch_num + fold_num * args.num_epochs)

            if args.k_folds:
                # Create predicted cross-section and enface images at the end of every fold.
                generate_inferred_images(EXP_DIR, model_state)
                print('Generated inferred images for fold {}'.format(fold_num))
                model_state.global_index = 0    # Reset counter used for Tensorboard scalar logging
            else:
                # The user can run our program separately in test/predict mode
                # with their testing eye sets if they wish to see the predicted en-face.
                print('Used full input data set for training. '
                      'No predictions generated.')

    else:   # Testing Mode (i.e. Just generate predicted images, no training)

        # load from latest checkpoint and load data for just 1 of 5 folds
        assert args.ckptdir is not None
        model_state = ModelState(
            is_training_mode=False, ckpt_dir=args.ckptdir, dataset=ds)
        print('Restoring from checkpoint at {}'.format(args.ckptdir))
        model_state.restore_from_checkpoint()

        # generate results based on prediction
        generate_inferred_images(EXP_DIR, model_state)
