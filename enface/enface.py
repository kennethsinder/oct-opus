from os.path import join

from configs.parameters import START_ROW, END_ROW, EXPERIMENT
from datasets.train_and_test import TESTING_DATASETS
from enface.image_io import ImageIO
from enface.slicer import Slicer

MULTI_SLICE_MAX_NORM = "multi_slice_max_norm.png"
MULTI_SLICE_SUM = "multi_slice_sum.png"


def gen_single_enface(predicted_dir, dataset, epoch_num):
    work_dir = join(predicted_dir, dataset)
    image_io = ImageIO()
    eye = image_io.load_single_eye(work_dir)
    slicer = Slicer()

    multi_slice_max_norm = slicer.multi_slice_max_norm(eye=eye, lower=START_ROW, upper=END_ROW)
    image_io.save_enface_image(enface=multi_slice_max_norm, filepath=work_dir, filename=MULTI_SLICE_MAX_NORM)
    EXPERIMENT.log_asset(
        file_data=join(work_dir, MULTI_SLICE_MAX_NORM),
        file_name="{}_epoch{}_{}".format(dataset, epoch_num, MULTI_SLICE_MAX_NORM),
        step=epoch_num
    )

    multi_slice_sum = slicer.multi_slice_sum(eye=eye, lower=START_ROW, upper=END_ROW)
    image_io.save_enface_image(enface=multi_slice_sum, filepath=work_dir, filename=MULTI_SLICE_SUM)
    EXPERIMENT.log_asset(
        file_data=join(work_dir, MULTI_SLICE_SUM),
        file_name="{}_epoch{}_{}".format(dataset, epoch_num, MULTI_SLICE_SUM),
        step=epoch_num
    )


def gen_enface_all_testing(predicted_dir, epoch_num):
    for test_dataset in TESTING_DATASETS:
        gen_single_enface(
            predicted_dir=predicted_dir,
            dataset=test_dataset,
            epoch_num=epoch_num
        )
