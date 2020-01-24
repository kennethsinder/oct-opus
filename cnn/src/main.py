import argparse
from datetime import datetime

import utils
from model import CNN
from params import TRAINING_DATASET_PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'], help='Specify the mode in which to run the mode')
    parser.add_argument('-e', '--num-epochs', type=int, default=10, help='Specify the number of epochs to train for')
    args = parser.parse_args()

    model = CNN()

    if args.mode == 'train':
       train_dataset = utils.get_dataset(TRAINING_DATASET_PATH)
       print(train_dataset)
       # history = model.train(train_dataset, args.num_epochs)
       # print(history)
       # model.save('./states/' + datetime.strftime('%d-%m-%Y_%H:%M:%S'))

    #test_dataset = utils.get_dataset('/private/fydp1/testing-data')
    #model.test(test_dataset)
    #print('Test loss:', loss)
