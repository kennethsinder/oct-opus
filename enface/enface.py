import traceback

from enface.image_io import ImageIO, MULTI_SLICE_SUM, MULTI_SLICE_MAX_NORM
from enface.slicer import Slicer


def gen_single_enface(dataset_dir,
                      normalize=False,
                      output_dir=None,
                      output_prefix=""):
    # image constants
    IMAGE_DIM = 512
    START_ROW = 50
    END_ROW = 256

    if output_dir is None:
        output_dir = dataset_dir

    if output_prefix != "":
        output_prefix = f"{output_prefix}_"

    try:
        image_io = ImageIO(IMAGE_DIM=IMAGE_DIM)

        eye = image_io.load_single_eye(dataset_dir)

        slicer = Slicer(IMAGE_DIM=IMAGE_DIM)

        # max norm
        filename = f"{output_prefix}{MULTI_SLICE_MAX_NORM}"
        multi_slice_max_norm = slicer.multi_slice_max_norm(
            eye=eye, lower=START_ROW, upper=END_ROW)
        image_io.save_enface_image(
            enface=multi_slice_max_norm,
            filepath=output_dir,
            filename=filename,
            normalize=normalize)
        print(f"Generated {filename}")

        # sum
        filename = f"{output_prefix}{MULTI_SLICE_SUM}"
        multi_slice_sum = slicer.multi_slice_sum(
            eye=eye, lower=START_ROW, upper=END_ROW)
        image_io.save_enface_image(
            enface=multi_slice_sum,
            filepath=output_dir,
            filename=filename,
            normalize=normalize)
        print(f"Generated {filename}")

    except FileNotFoundError:
        # Case where `dataset_dir` does not contain any images from which we
        # can create an enface. In this case, multi_slice_max_norm.png and
        # multi_slice_sum.png are not generated.
        traceback.print_exc()  # debugging purposes
        exit(1)
