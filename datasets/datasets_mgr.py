from glob import glob
from os.path import join

from PIL import Image

from datasets.all_datasets import ALL_DATASETS
from configs.parameters import ALL_DATA_DIR, BSCAN_DIRNAME, OMAG_DIRNAME, IMAGE_DIM


class DatasetsManager:
    @staticmethod
    def check_image_dimensions():
        for dataset in ALL_DATASETS:
            for file_type in [BSCAN_DIRNAME, OMAG_DIRNAME]:
                for image_file in glob(join(ALL_DATA_DIR, dataset, file_type, "*.png")):
                    img_path = join(ALL_DATA_DIR, dataset, file_type, image_file)
                    with Image.open(img_path) as img:
                        width, height = img.size
                        if width != IMAGE_DIM or height != IMAGE_DIM:
                            print("Image " + img_path + " has dimensions (" + str(width) + "x" + str(height) + ")")


if __name__ == '__main__':
    dsm = DatasetsManager()
    dsm.check_image_dimensions()
