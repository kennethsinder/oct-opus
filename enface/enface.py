import traceback

from os.path import join

from cgan.parameters import START_ROW, END_ROW, EXPERIMENT
from enface.image_io import ImageIO
from enface.slicer import Slicer

MULTI_SLICE_MAX_NORM = "multi_slice_max_norm.png"
MULTI_SLICE_SUM = "multi_slice_sum.png"


def gen_single_enface(predicted_dir, dataset, epoch_num):
    work_dir = join(predicted_dir, dataset)
    image_io = ImageIO()
    try:
        eye = image_io.load_single_eye(work_dir)
    except FileNotFoundError:
        # Case where `work_dir` does not contain any
        # images from which we can create an enface. In this case
        # just don't create the multi_slice_max_norm.png and
        # multi_slice_sum.png enfaces and silently continue.
        traceback.print_exc()  # So we can diagnose why it's empty
        return
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


def gen_enface_all_testing(predicted_dir, epoch_num, datasets):
    for test_dataset in datasets[1]:
        gen_single_enface(
            predicted_dir=predicted_dir,
            dataset=test_dataset,
            epoch_num=epoch_num
        )


if __name__ == '__main__':
    """
    Executing this file directly via CLI will generate enfaces
    for every eye folder in the path specified by joining together
    the two command-line arguments. Enfaces are saved in each eye folder
    images were sourced from.
    Example Usage:  ./enface.py ./RUN_11 predicted-epoch-5
    """
    import sys
    from glob import glob

    for f in glob(join(sys.argv[1], sys.argv[2], '*')):
        gen_single_enface(f, 'OMAG Bscans', 0)
