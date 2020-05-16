import random
from os import listdir
from os.path import join, isdir
from typing import List, Tuple, Set


class Dataset:

    def __init__(self, root_data_path, num_folds=1, seed=42):
        self.root_data_path: str = root_data_path
        self.num_folds: int = num_folds

        ls = set(listdir(root_data_path))
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
        if self.num_folds == 1:
            return self.__all_datasets, set()

        test_sets = self.__folds[fold_id]
        train_sets = self.__all_datasets - test_sets
        return train_sets, test_sets

    def get_all_datasets(self) -> Set:
        return self.__all_datasets.copy()
