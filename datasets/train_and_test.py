from datasets.all_datasets import ALL_DATASETS
from random import sample

USABLE_DATASETS = {
    "2015-10-21___512_2048_Horizontal_Images50",
    "2015-10-21___512_2048_Horizontal_Images58",
    "2015-10-21___512_2048_Horizontal_Images85",
    "2015-10-21___512_2048_Horizontal_Images89",
    "2015-10-21___512_2048_Horizontal_Images90",
    "2015-10-22___512_2048_Horizontal_Images13",
    "2015-10-22___512_2048_Horizontal_Images16",
    "2015-10-22___512_2048_Horizontal_Images2",
    "2015-10-22___512_2048_Horizontal_Images20",
    "2015-10-22___512_2048_Horizontal_Images34",
    "2015-10-22___512_2048_Horizontal_Images37",
    "2015-10-22___512_2048_Horizontal_Images41",
    "2015-10-22___512_2048_Horizontal_Images58",
    "2015-10-22___512_2048_Horizontal_Images63",
    "2015-10-22___512_2048_Horizontal_Images71",
    "2015-10-22___512_2048_Horizontal_Images8",
    "2015-10-23___512_2048_Horizontal_Images14",
    "2015-10-23___512_2048_Horizontal_Images19",
    "2015-10-23___512_2048_Horizontal_Images2",
    "2015-10-23___512_2048_Horizontal_Images21",
    "2015-10-23___512_2048_Horizontal_Images32",
    "2015-10-23___512_2048_Horizontal_Images37",
    "2015-10-23___512_2048_Horizontal_Images58",
    "2015-10-23___512_2048_Horizontal_Images64",
    "2015-10-23___512_2048_Horizontal_Images9",
    "2015-10-26___512_2048_Horizontal_Images10",
    "2015-10-26___512_2048_Horizontal_Images19",
    "2015-10-26___512_2048_Horizontal_Images24",
    "2015-10-26___512_2048_Horizontal_Images28",
    "2015-10-26___512_2048_Horizontal_Images31",
    "2015-10-26___512_2048_Horizontal_Images35",
    "2015-10-26___512_2048_Horizontal_Images43",
    "2015-10-26___512_2048_Horizontal_Images47",
    "2015-10-26___512_2048_Horizontal_Images5",
    "2015-10-26___512_2048_Horizontal_Images50",
    "2015-10-26___512_2048_Horizontal_Images60",
    "2015-10-26___512_2048_Horizontal_Images68",
    "2015-10-26___512_2048_Horizontal_Images73",
    "2015-10-26___512_2048_Horizontal_Images74",
    "2015-10-27___512_2048_Horizontal_Images11",
    "2015-10-27___512_2048_Horizontal_Images17",
    "2015-10-27___512_2048_Horizontal_Images22",
    "2015-10-27___512_2048_Horizontal_Images27",
    "2015-10-27___512_2048_Horizontal_Images34",
    "2015-10-27___512_2048_Horizontal_Images39",
    "2015-10-27___512_2048_Horizontal_Images40",
    "2015-10-27___512_2048_Horizontal_Images58",
    "2015-10-27___512_2048_Horizontal_Images6",
    "2015-10-27___512_2048_Horizontal_Images67",
    "2015-10-27___512_2048_Horizontal_Images72",
    "2015-10-27___512_2048_Horizontal_Images73",
    "2015-10-20___512_2048_Horizontal_Images15",
    "2015-10-20___512_2048_Horizontal_Images21",
    "2015-10-20___512_2048_Horizontal_Images27",
    "2015-10-20___512_2048_Horizontal_Images33",
    "2015-10-20___512_2048_Horizontal_Images37",
    "2015-10-20___512_2048_Horizontal_Images43",
    "2015-10-20___512_2048_Horizontal_Images54",
    "2015-10-20___512_2048_Horizontal_Images83",
    "2015-10-20___512_2048_Horizontal_Images99",
    "2015-10-21___512_2048_Horizontal_Images100",
    "2015-10-21___512_2048_Horizontal_Images21",
    "2015-10-21___512_2048_Horizontal_Images26",
    "2015-10-21___512_2048_Horizontal_Images31",
    "2015-10-21___512_2048_Horizontal_Images4",
    "2015-10-21___512_2048_Horizontal_Images41",
}

usable_datasets_copy = set(USABLE_DATASETS)

K = 5
folds = []
for i in range(K - 1):
    folds.append(set(sample(usable_datasets_copy,
                            min(len(usable_datasets_copy), int(len(USABLE_DATASETS) / K)))))
    usable_datasets_copy = usable_datasets_copy - folds[i]
folds.append(usable_datasets_copy)


def train_and_test_sets(i):
    test_sets = folds[i]
    train_sets = USABLE_DATASETS - test_sets
    return train_sets, test_sets


assert USABLE_DATASETS.issubset(ALL_DATASETS)
