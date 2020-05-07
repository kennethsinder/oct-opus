import glob
import random
from os import listdir
from os.path import join, isdir
from typing import List, Tuple, Set

from cgan.parameters import OMAG_DIRNAME, BSCAN_DIRNAME


class Dataset:

    def __init__(self, root_data_path, num_folds=1, seed=42):
        self.__root_data_path: str = root_data_path
        temp = set(listdir(root_data_path))
        ls = map(lambda x: join(root_data_path, x), temp)
        # save only the directories
        self.__all_datasets: Set = set(filter(lambda x: isdir(join(root_data_path, x)), ls))
        assert len(self.__all_datasets) > 0
        print("Found {} datasets under {}".format(len(self.__all_datasets), root_data_path))

        # sets the seed for random
        random.seed(seed)

        # partitions the dataset into folds
        self.__folds: List[Set] = []
        usable_datasets_copy = self.__all_datasets.copy()
        for i in range(num_folds - 1):
            self.__folds.append(set(
                random.sample(
                    usable_datasets_copy,
                    min(len(usable_datasets_copy), int(len(self.__all_datasets) / num_folds))
                )
            ))
            usable_datasets_copy = usable_datasets_copy - self.__folds[i]
        self.__folds.append(usable_datasets_copy)

    def get_train_and_test_by_fold_id(self, fold_id) -> Tuple[Set, Set]:
        test_sets = self.__folds[fold_id]
        train_sets = self.__all_datasets - test_sets
        return train_sets, test_sets

    def get_all_datasets(self) -> Set:
        return self.__all_datasets.copy()

    def get_all_folds(self) -> List[Set]:
        return self.__folds.copy()

    def get_all_bscans(self) -> List:
        __image_files = []
        for dataset_path in self.__all_datasets:
            __image_files.extend(glob.glob(join(dataset_path, BSCAN_DIRNAME, '[0-9]*.png')))
        assert len(__image_files) > 0
        return __image_files

    def get_all_omags(self) -> List:
        __image_files = []
        for dataset_path in self.__all_datasets:
            __image_files.extend(glob.glob(join(dataset_path, OMAG_DIRNAME, '[0-9]*.png')))
        assert len(__image_files) > 0
        return __image_files
