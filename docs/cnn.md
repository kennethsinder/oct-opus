# CNN

The CNN (convolutional neural network) found in <code>[/cnn](../cnn)</code> is an implementation of the U-net model described in 'Generating retinal flow maps from structural optical coherence tomography with artificial intelligence' by Lee et al. The paper can be found here: https://doi.org/10.1038/s41598-019-42042-y

We thank Dr. Aaron Y. Lee (leeay@uw.edu) for providing the code for the U-net model.

This CNN implementation exists because we wanted to compare the performance of the cGAN in <code>[/cgan](../cgan)</code> to a baseline/state-of-the-art. There should no longer be any reason to run this code. However, an overview of training/testing and enface generation can be found below.

## Entrypoint

The entrypoint for the CNN is <code>[cnn.py](../cnn.py)</code>. There are many more arguments for the CNN than the cGAN; more detail on each argument can be found by running
```
python cnn.py -h
```

For each time <code>[cnn.py](../cnn.py)</code> is run, the data (e.g. checkpoints) is saved in the directory specified by `-ex/--experiment-dir`. The experiment dir has the following structure:
```
experiment_dir/
├── checkpoints
│   ├── checkpoints
│   ├── ckpt-1.index
│   ├── ckpt-1.data-00000-of-00002
│   ├── ckpt-1.data-00001-of-00002
│   └── ...
├── cross_sections
│   ├── epoch_1.png
│   ├── epoch_2.png
│   └── ...
├── enfaces
│   └── epoch_5
│       └── ...
└── logs-XX-XX-XXXX_XXhXXmXXs
    └── ...
```
To load the model found in `experiment_dir`, either to generate enfaces or to continue training, run
```
python cnn.py <mode> <hardware <augment_level> -ex ~/experiment_dir ...
```
The latest checkpoint from `experiment_dir/checkpoints` will be loaded. Furthermore, a new Tensorboard `logs-XX-XX-XXXX_XXhXXmXXs` directory will be generated.

There are 4 required arguments for <code>[cnn.py](../cnn.py)</code>:
- `mode`
    - `train`: Train and test the model
    - `enface`: Generate an enface, using the model located in the experiment dir. If no dir was specified, an untrained model will be used.
- `hardware`
    - `cpu`/`gpu`
    - Depends on the computer you are running the program on
- `augment_level`
    - Specifies what level of data augmentation should be done on the data.
    - `normalize`: Normalize the training and testing B-scans using the mean and standard deviation of the training B-scans. The mean and s.t.d. must be calculated beforehand and stored in <code>[parameters.py](../cnn/parameters.py)</code>. You can use <code>[calculate_mean_and_std.py](../scripts/cnn/calculate_mean_and_std.py)</code> to calculate the values.
    - `contrast`: Increase the contrast of the B-scans and the OMAG images.
    - `full_augment`: Perform the same augmentations as the cGAN.
- `-d/--data-dir`.

There also exists <code>[run_cnn_job.sh](../run_cnn_job.sh)</code>, which is a wrapper of <code>[cnn.py](../cnn.py)</code> for the Sharcnet cluster.

The CNN accepts the same data as the cGAN, organized in the same manner (see <code>[training_and_testing.md](training_and_testing.md)</code> for more information).

## Training and Testing

Like the cGAN, the CNN performs k-folds cross validation. Unlike the cGAN, which trains on each fold sequentially, the CNN is designed to train each fold in parallel. This is done using the `--k-folds/-k` and `-selected-fold/-s` arguments. For example, if you want to train the first of 5 folds, you would run
```
python cnn.py train <hardware> <augment_level> -k 5 -s 0 ...
```
To run the second fold, you run
```
python cnn.py train <hardware> <augment_level> -k 5 -s 1 ...
```
and so on.
However, doing this will cause each fold to train with different initial weights. If you want each fold to start with the same initial weights (as the cGAN does), you must first 'seed' each experiment dir (`-ex/--experiment-dir`) with the desired initial weights. That is, you must:

1. Generate some initial weights for the CNN. You can do this by creating a CNN object using the desired parameters, then immediately saving a checkpoint.
```
checkpoints
ckpt-1.index
ckpt-1.data-00000-of-00002
ckpt-1.data-00001-of-00002
```

2. Create experiment directories
```
experiments/
├── fold_0
├── fold_1
├── ...
└── fold_n
```

3. Insert the initial weights into each experiment dir
```
experiments/
├── fold_0
│   └── checkpoints
│       ├── checkpoints
│       ├── ckpt-1.index
│       ├── ckpt-1.data-00000-of-00002
│       └── ckpt-1.data-00001-of-00002
├── fold_1
│   └── checkpoints
│       ├── checkpoints
│       ├── ckpt-1.index
│       └── ...
├── ...
└── fold_n
    └── checkpoints
        ├── checkpoints
        ├── ckpt-1.index
        └── ...
```

4. Run each fold, specifying the experiment dir
```
python cnn.py train <hardware> <augment_level> -k n -s 0 -ex ~/experiments/fold_0 ...
python cnn.py train <hardware> <augment_level> -k n -s 1 -ex ~/experiments/fold_1 ...
...
```

> :warning: This will mean the checkpoint names do not align with the epoch numbers. For example, after training for 1 epoch, the model weights are saved in `ckpt-2.*`, rather than the expected `ckpt-1.*`.

After training for the specified number of epochs, the script will generate enfaces using the latest model weights. The enfaces are stored in the experiment dir, under the `enface` dir.

## Enface Generation

In `enface` mode, in addition to all the other regular arguments, the argument `-ef/--enface-dir` must also be specified.
```
python cnn.py enface <hardware> <augment_level> -ef eye_data -k n -s 0 -ex ~/experiments/fold_0  ...
```
Where `eye_data` is organized as
```
eye_data
├── Capillary\ Red\ Layer.tiff
├── OMAG\ Bscans
└── xzIntensity
```

The generated enfaces are stored in the experiment dir, under the `enface` dir.
