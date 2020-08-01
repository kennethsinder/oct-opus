import argparse
import concurrent.futures
import datetime
import json
import os
import random
import sys
import time

import tensorflow as tf

from typing import Any, Dict, Set

from cgan.model_state import ModelState
from cgan.parameters import GPU, LAYER_BATCH
from cgan.utils import generate_inferred_enface, get_dataset

# https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')

PARAMS_FILE = 'experiment_parameters.json'


def save_exp_params(exp_params: Dict[str, Any]) -> None:
    with open(os.path.join(exp_params["name"], PARAMS_FILE), "w") as outfile:
        json.dump(exp_params, outfile, indent=4)


def load_exp_params(exp_dir: str) -> Dict[str, Any]:
    with open(os.path.join(exp_dir, PARAMS_FILE), "r") as infile:
        return json.load(infile)


def get_dataset_names(root_data_path: str) -> Set[str]:
    ls = set(os.listdir(root_data_path))

    # save only the directories
    all_datasets: Set[str] = set(
        filter(lambda x: os.path.isdir(os.path.join(root_data_path, x)), ls))
    assert len(all_datasets) > 0
    print("Found {} datasets under {}".format(len(all_datasets),
                                              root_data_path))

    return all_datasets


def train_models(data_dir: str,
                 dataset_names: Set[str],
                 models: Dict[str, ModelState],
                 epochs: int,
                 start_epoch: int = 1) -> None:
    for model in models.values():
        datasets = get_dataset(data_dir, dataset_names - model.holdout_set)
        if epochs < model.epoch.numpy():
            print(f"----- {model.name} already trained to {epochs} epochs, skipping -----")
            continue

        for i in range(start_epoch, epochs + 1):
            if i < model.epoch.numpy():
                continue

            print(f"----- Starting epoch number {i} for {model.name} -----")
            start = time.time()

            for dataset_name, images in datasets:
                model.train_step(images[0], images[1])

            print(f"----- Epoch {i} completed for {model.name} in {time.time() - start} seconds -----")
            model.end_epoch_and_checkpoint()
            print(f"----- Checkpointed {model.name} after epoch {i} -----")

        for dataset_name in dataset_names:
            if dataset_name not in model.holdout_set:
                continue

            generate_inferred_enface(
                os.path.join(data_dir, dataset_name),
                model.output_dir,
                model.generator
            )

        print(f"----- Generating predictions for {model.name} -----")
        print(f"----- Generated predictions for {model.name} -----")


def generate_predictions(data_dir: str,
                         dataset_names:
                         Set[str],
                         models: Dict[str, ModelState]) -> None:
    for dataset_name in dataset_names:
        for model in models.values():
            if dataset_name not in model.holdout_set:
                continue

            generate_inferred_enface(
                os.path.join(data_dir, dataset_name),
                model.output_dir,
                model.generator
            )


def run_predict(name: str, args: argparse.Namespace) -> None:
    prev_exp_params = load_exp_params(args.experiment)

    exp_params = prev_exp_params.copy()
    exp_params["name"] = name
    exp_params["mode"] = "predict"
    exp_params["layer_batch"] = LAYER_BATCH
    exp_params["loaded_exp"] = os.path.abspath(args.experiment)
    exp_params["data"] = os.path.abspath(args.data)

    all_dataset_names: Set[str] = get_dataset_names(args.data)
    exp_params["datasets"] = list(all_dataset_names)

    if args.model is not None:
        exp_params["models"] = {
            args.model: prev_exp_params["models"][args.model]
        }

    models = dict()
    for model_name, model_params in exp_params["models"].items():
        models[model_name] = ModelState(
            name=model_name,
            exp_dir=exp_params["name"],
            data_dir=exp_params["data"],
            holdout_set=set(model_params["holdout"])
        )
        models[model_name].restore_from_checkpoint(
            exp_params["loaded_exp"], predict_only=True
        )

    save_exp_params(exp_params)

    generate_predictions(exp_params["data"],
                         all_dataset_names,
                         models)


def run_repeat(name: str, args: argparse.Namespace) -> None:
    prev_exp_params = load_exp_params(args.experiment)

    exp_params = prev_exp_params.copy()
    exp_params["name"] = name
    exp_params["loaded_exp"] = os.path.abspath(args.experiment)
    exp_params["layer_batch"] = LAYER_BATCH
    exp_params["data"] = os.path.abspath(args.data)

    all_dataset_names: Set[str] = get_dataset_names(args.data)
    exp_params["datasets"] = list(all_dataset_names)

    if args.epochs is not None:
        exp_params["epochs"] = args.epochs

    models = dict()
    for model_name, model_params in exp_params["models"].items():
        models[model_name] = ModelState(
            name=model_name,
            exp_dir=exp_params["name"],
            data_dir=exp_params["data"],
            holdout_set=set(model_params["holdout"])
        )

    save_exp_params(exp_params)

    train_models(exp_params["data"],
                 all_dataset_names,
                 models,
                 exp_params["epochs"])

    generate_predictions(exp_params["data"],
                         all_dataset_names,
                         models)


def run_continue(name: str, args: argparse.Namespace) -> None:
    prev_exp_params = load_exp_params(args.experiment)

    exp_params = prev_exp_params.copy()
    exp_params["name"] = name
    exp_params["loaded_exp"] = os.path.abspath(args.experiment)
    exp_params["layer_batch"] = LAYER_BATCH
    exp_params["data"] = os.path.abspath(args.data)

    all_dataset_names: Set[str] = get_dataset_names(args.data)
    exp_params["datasets"] = list(all_dataset_names)

    if args.epochs is not None:
        exp_params["epochs"] = args.epochs

    models = dict()
    for model_name, model_params in exp_params["models"].items():
        models[model_name] = ModelState(
            name=model_name,
            exp_dir=exp_params["name"],
            data_dir=exp_params["data"],
            holdout_set=set(model_params["holdout"])
        )
        models[model_name].restore_from_checkpoint(
            exp_params["loaded_exp"], predict_only=False
        )

    save_exp_params(exp_params)

    train_models(exp_params["data"],
                 all_dataset_names,
                 models,
                 exp_params["epochs"])

    generate_predictions(exp_params["data"],
                         all_dataset_names,
                         models)


def run_k_folds(name: str, args: argparse.Namespace) -> None:
    exp_params = {
        "name": name,
        "mode": "k_folds",
        "layer_batch": LAYER_BATCH,
        "epochs": args.epochs,
        "data": os.path.abspath(args.data),
    }

    if args.num_folds < 2:
        print("Error: k-folds cross-validation requires multiple folds")
        sys.exit(1)

    all_dataset_names: Set[str] = get_dataset_names(args.data)
    exp_params["datasets"] = list(all_dataset_names)

    models = dict()
    exp_params["models"] = dict()

    # divide the datasets into k holdout sets, and create a
    # model instance per holdout set
    remaining_dataset_names = all_dataset_names.copy()
    for i in range(1, args.num_folds + 1):
        holdout: Set[str] = set(
            random.sample(
                remaining_dataset_names,
                min(len(remaining_dataset_names),
                    len(all_dataset_names) // args.num_folds)))

        model_name = f"fold_{i}"
        models[model_name] = ModelState(
            name=model_name,
            exp_dir=exp_params["name"],
            data_dir=exp_params["data"],
            holdout_set=holdout)

        exp_params["models"][model_name] = dict()
        exp_params["models"][model_name]["holdout"] = list(holdout)

        remaining_dataset_names = remaining_dataset_names - holdout

    save_exp_params(exp_params)

    train_models(exp_params["data"],
                 all_dataset_names,
                 models,
                 args.epochs)

    generate_predictions(exp_params["data"],
                         all_dataset_names,
                         models)


def run_train_test(name: str, args: argparse.Namespace) -> None:
    exp_params = {
        "name": name,
        "mode": "train_test",
        "layer_batch": LAYER_BATCH,
        "epochs": args.epochs,
        "data": os.path.abspath(args.data),
    }

    if args.test_set_size < 0 or args.test_set_size > 100:
        print("Error: test split must be a percentage between 0 and 100")
        sys.exit(1)

    all_dataset_names: Set[str] = get_dataset_names(args.data)
    exp_params["datasets"] = list(all_dataset_names)

    models = dict()
    exp_params["models"] = dict()

    holdout: Set[str] = set(
        random.sample(
            all_dataset_names,
            min(len(all_dataset_names),
                int(len(all_dataset_names) * args.test_set_size / 100))))

    model_name = "train_test"
    models[model_name] = ModelState(
        name=model_name,
        exp_dir=exp_params["name"],
        data_dir=exp_params["data"],
        holdout_set=holdout)

    exp_params["models"][model_name] = dict()
    exp_params["models"][model_name]["holdout"] = list(holdout)

    save_exp_params(exp_params)

    train_models(exp_params["data"],
                 all_dataset_names,
                 models,
                 args.epochs)

    generate_predictions(exp_params["data"],
                         all_dataset_names,
                         models)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or use the BScan to OMAG style transfer CGAN.")
    subparsers = parser.add_subparsers()

    continue_parser = subparsers.add_parser(
        "continue",
        help="create a new experiment that continues from "
        "the last checkpoint in an existing experiment")
    continue_parser.add_argument("experiment",
                                 type=str,
                                 help="experiment directory containing model "
                                 "to continue training")
    continue_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="target number of training epochs for the "
        "final model: includes the number of epochs "
        "for which the model has already been trained "
        "[default: inherited from existing experiment]")
    continue_parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="data directory [default: inherited from existing experiment]")
    continue_parser.set_defaults(func=run_continue)

    k_folds_parser = subparsers.add_parser(
        "k-folds",
        help="train k different models using different folds "
        "of data to test each time; generate predicted images from the "
        "test sets after training")
    k_folds_parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="data directory")
    k_folds_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=3,
        help="number of training epochs for each model "
        "[default: 3]")
    k_folds_parser.add_argument(
        "-k",
        "--num-folds",
        type=int,
        default=5,
        help="number of folds into which to divide the data "
        "[default: 5]")
    k_folds_parser.set_defaults(func=run_k_folds)

    train_test_parser = subparsers.add_parser(
        "train-test",
        help="train a model on a subset of the data, "
        "retaining some for testing; generate predicted images from the "
        "test set after training")
    train_test_parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="data directory")
    train_test_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=3,
        help="number of training epochs [default: 3]")
    train_test_parser.add_argument(
        "-s",
        "--test-set-size",
        type=int,
        default=20,
        help="percent of the data to reserve for testing "
        "[default: 20]")
    train_test_parser.set_defaults(func=run_train_test)

    predict_parser = subparsers.add_parser("predict", help="")
    predict_parser.add_argument(
        "experiment",
        type=str,
        help="experiment directory containing model "
        "to use to generate predictions")
    predict_parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="data directory")
    predict_parser.add_argument(
        "-m", "--model",
        type=str,
        help="name of the specific model to use to predict data "
        "[default: all]")
    predict_parser.set_defaults(func=run_predict)

    repeat_parser = subparsers.add_parser("repeat", help="")
    repeat_parser.add_argument(
        "experiment",
        type=str,
        help="experiment directory containing model "
        "to use to generate repeations")
    repeat_parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="data directory")
    repeat_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="number of training epochs "
        "[default: inherited from existing experiment]")
    repeat_parser.set_defaults(func=run_repeat)

    return parser.parse_args()


if __name__ == "__main__":
    args: argparse.Namespace = get_args()

    device_name = tf.test.gpu_device_name()
    if device_name != GPU:
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    exp_name: str = "experiment-{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))

    # Main directory used to store output
    os.makedirs(exp_name, exist_ok=False)

    with open(os.path.join(exp_name, "README.md"),
              "w") as readme_file:
        # Create a mostly blank README file to encourage good
        # documentation of the purpose of each experiment.
        readme_file.write(f"# {exp_name}\n\n")

    # Delegate to specific subcommand
    args.func(exp_name, args)
