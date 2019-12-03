from configs.parameters import IS_ENFACE_TRAINING

if IS_ENFACE_TRAINING:
    TRAINING_DATASETS = {'normalized_train', }
    TESTING_DATASETS = {'normalized_test', }
else:
    TRAINING_DATASETS = {
        "2015-10-23___512_2048_Horizontal_Images9",
        "2015-10-23___512_2048_Horizontal_Images14",
        "2015-10-23___512_2048_Horizontal_Images32",
        "2015-10-27___512_2048_Horizontal_Images67",
    }

    TESTING_DATASETS = {
        "2015-10-23___512_2048_Horizontal_Images64",
        "2015-10-23___512_2048_Horizontal_Images37",
        "2015-10-22___512_2048_Horizontal_Images41",
    }
