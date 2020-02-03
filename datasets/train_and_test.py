from datasets.all_datasets import ALL_DATASETS

TRAINING_DATASETS = {
    "2015-10-21___512_2048_Horizontal_Images50",
}

TESTING_DATASETS = {
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

assert TRAINING_DATASETS.issubset(ALL_DATASETS)
assert TESTING_DATASETS.issubset(ALL_DATASETS)
assert TESTING_DATASETS.isdisjoint(TRAINING_DATASETS)
