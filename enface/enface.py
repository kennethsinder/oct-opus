from os.path import join

from configs.parameters import START_ROW, END_ROW
from datasets.train_and_test import TESTING_DATASETS
from enface.image_io import ImageIO
from enface.slicer import Slicer


def gen_single_enface(input_dir, input_type: ImageIO.InputType, output_dir):
    image_io = ImageIO(input_dir, input_type)
    eye = image_io.load_single_eye()
    slicer = Slicer()

    multi_slice_max_norm = slicer.multi_slice_max_norm(eye=eye, lower=START_ROW, upper=END_ROW)
    image_io.save_enface_image(enface=multi_slice_max_norm, filepath=output_dir, filename="multi_slice_max_norm.png")

    multi_slice_sum = slicer.multi_slice_sum(eye=eye, lower=START_ROW, upper=END_ROW)
    image_io.save_enface_image(enface=multi_slice_sum, filepath=output_dir, filename="multi_slice_sum.png")


def gen_enface_all_testing(predicted_dir):
    for test_dataset in TESTING_DATASETS:
        gen_single_enface(
            input_dir=join(predicted_dir, test_dataset),
            input_type=ImageIO.InputType.OMAG,
            output_dir=join(predicted_dir, test_dataset)
        )
