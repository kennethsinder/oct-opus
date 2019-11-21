from enum import Enum


class DatasetsManager:

    class DatasetType(Enum):
        TRAINING = 1
        TESTING = 2
        IGNORE = 3

    def __init__(self):
        self.__training_set = {
            "2015-07-13-Images123",
            "2015-07-13-Images132",
            "2015-07-13-Images20",
            "2015-07-13-Images73",
            "2015-07-13-Images78",
            "2015-07-14-Images110",
            "2015-07-14-Images117",
            "2015-07-14-Images12",
            "2015-07-14-Images24",
            "2015-07-14-Images64",
            "2015-07-14-Images70"
        }

        self.__testing_set = {
            "2015-07-27-Images100",
            "2015-07-27-Images13-0",
            "2015-07-27-Images50",
            "2015-07-27-Images61",
            "2015-07-27-Images8",
            "2015-07-27-Images91",
        }

    def query_dataset_type(self, dirname) -> DatasetType:
        if dirname in self.__training_set:
            return self.DatasetType.TRAINING
        if dirname in self.__testing_set:
            return self.DatasetType.TESTING
        return self.DatasetType.IGNORE

    def get_training_set_names(self):
        return self.__training_set.copy()

    def get_testing_set_name(self):
        return self.__testing_set.copy()
