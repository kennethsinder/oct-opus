import traceback

from enface.image_io import ImageIO, MULTI_SLICE_SUM, MULTI_SLICE_MAX_NORM
from enface.slicer import Slicer


def gen_single_enface(dataset_dir):
    # image constants
    IMAGE_DIM = 512
    START_ROW = 50
    END_ROW = 256

    try:
        image_io = ImageIO(IMAGE_DIM=IMAGE_DIM)
        eye = image_io.load_single_eye(dataset_dir)

        slicer = Slicer(IMAGE_DIM=IMAGE_DIM)

        # max norm
        multi_slice_max_norm = slicer.multi_slice_max_norm(eye=eye, lower=START_ROW, upper=END_ROW)
        image_io.save_enface_image(enface=multi_slice_max_norm, filepath=dataset_dir, filename=MULTI_SLICE_MAX_NORM)
        print("Generated", MULTI_SLICE_MAX_NORM)

        # sum
        multi_slice_sum = slicer.multi_slice_sum(eye=eye, lower=START_ROW, upper=END_ROW)
        image_io.save_enface_image(enface=multi_slice_sum, filepath=dataset_dir, filename=MULTI_SLICE_SUM)
        print("Generated", MULTI_SLICE_SUM)

    except FileNotFoundError:
        # Case where `dataset_dir` does not contain any images from which we can create an enface.
        # In this case, multi_slice_max_norm.png and multi_slice_sum.png are not generated.
        traceback.print_exc()  # debugging purposes
        exit(1)
