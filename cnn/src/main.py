import utils
from model import CNN
from datetime import datetime


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'], help='Specify the mode in which to run the mode')
    parser.add_argument('-l', '--load', help='Load model state from the specified filepath')
    parser.add_argument('-e', '--num-epochs', type=int, default=10, help='Specify the number of epochs to train for')
    return parser.parse_args()


if __name__ == '__main__':
    args = getArgs()

    model = CNN()
    if args.load:
        model.load(args.load)

    if args.mode == 'train':
        string_datetime = datetime.strftime('%d-%m-%Y_%H:%M:%S')
        writer = tf.summary.create_file_writer('./logs/' + string_datetime)
        train_dataset = utils.get_dataset('/private/fydp1/training-data')
        model.train(train_dataset, args.num_epochs)
        model.save('./states/' + string_datetime)

    test_dataset = utils.get_dataset('/private/fydp1/testing-data')
    model.test(test_dataset)
    print('Test loss:', loss)
