from datasets.all_datasets import ALL_DATASETS

TRAINING_DATASETS = {
    "2015-07-27 Images13-0",
    "2015-07-27 Images50",
    "2015-07-27 Images61",
    "2015-07-27 Images8",
    "2015-07-27 Images91",
    "2015-08-10 Images43-0",
    "2015-08-10 Images48-0",
    "2015-08-10 Images5-0",
    "2015-08-10 Images79-0",
    "2015-08-10 Images91-0",
    "2015-08-11 Images 50",
    "2015-08-11 Images10",
    "2015-08-11 Images45",
    "2015-08-11 Images5",
    "2015-08-11 Images77",
    "2015-08-11 Images84",
    "2015-09-07 Images13-0",
    "2015-09-07 Images41-0",
    "2015-09-07 Images46-0",
    "2015-09-07 Images73-0",
    "2015-09-07 Images78-0",
    "2015-09-07 Images8",
    "2015-09-07Images46-0",
    "2015-09-08 Images36-0",
    "2015-09-08 Images4-0",
    "2015-09-08 Images41-0",
    "2015-09-08 Images9-0",
}

TESTING_DATASETS = {
    "2015-07-14 Images110",
    "2015-07-14 Images117",
    "2015-07-14 Images12",
    "2015-07-14 Images24",
    "2015-07-14 Images64",
    "2015-07-14 Images70",
    "2015-07-27 Images100",
}

assert TRAINING_DATASETS.issubset(ALL_DATASETS)
assert TESTING_DATASETS.issubset(ALL_DATASETS)
assert TESTING_DATASETS.isdisjoint(TRAINING_DATASETS)
