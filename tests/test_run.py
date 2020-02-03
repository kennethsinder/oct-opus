import argparse
import os
import tensorflow as tf
import unittest
from unittest import mock
os.environ['UNITTEST'] = 'true'
from run import main


class TestRun(unittest.TestCase):
    def tearDown(self):
        del os.environ['UNITTEST']

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(starting_epoch=1,
                                                ending_epoch=5,
                                                mode='train',
                                                datadir='',
                                                hardware='cpu'))
    @mock.patch('src.train.train_epoch')
    @mock.patch('src.model_state.ModelState')
    @mock.patch('src.utils.generate_inferred_images')
    @mock.patch('src.utils.generate_cross_section_comparison')
    @mock.patch.dict(os.environ,{'UNITTEST': 'true'})
    def test_runs_correct_num_epochs(self, args_mock, epoch_mock, *args, **kwargs):
        main()
        self.assertEqual(epoch_mock.call_count, 5)
