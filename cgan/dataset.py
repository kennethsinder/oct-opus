import random
from os import listdir
from os.path import join, isdir


class Dataset:

    def __init__(self, root_data_path, num_folds=1, seed=42):
        self.__root_data_path = root_data_path
        ls = set(listdir(root_data_path))
        # save only the directories
        self.__all_datasets = set(filter(lambda x: isdir(join(root_data_path, x)), ls))
        assert len(self.__all_datasets) > 0

        # sets the seed for random
        random.seed(seed)

        # partitions the dataset into folds
        self.__folds = []
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

    def get_train_and_test_by_fold_id(self, fold_id):
        test_sets = self.__folds[fold_id]
        train_sets = self.__all_datasets - test_sets
        return train_sets, test_sets

    def get_all_datasets(self):
        return self.__all_datasets.copy()

    def get_all_folds(self):
        return self.__folds.copy()
