# Training and Testing

## Training

TDB

## Testing / Prediction

To be able to generate predicted B-Scans using a trained model simply call `cgan.py` via the following options.

```
python cgan.py predict -d {top_level_data_directory} -c {checkpoint_directory}
```

A more concrete example looks like this.

```
python cgan.py predict -d ../sample_datasets -c ./training_checkpoints
```

Note that `top_level_data_directory` should have the datasets nested like so.
Actual image files not shown for brevity.
```
sample_datasets/
├── 2015-10-27___512_2048_Horizontal_Images58
│   ├── Capillary\ Red\ Layer.tiff
│   ├── OMAG\ Bscans
│   └── xzIntensity
├── 2015-10-27___512_2048_Horizontal_Images6
│   ├── Capillary\ Red\ Layer.tiff
│   ├── OMAG\ Bscans
│   └── xzIntensity
├── 2015-10-27___512_2048_Horizontal_Images67
│   ├── Capillary\ Red\ Layer.tiff
│   ├── OMAG\ Bscans
│   └── xzIntensity
├── 2015-10-27___512_2048_Horizontal_Images72
│   ├── Capillary\ Red\ Layer.tiff
│   ├── OMAG\ Bscans
│   └── xzIntensity
└── 2015-10-27___512_2048_Horizontal_Images73
    ├── Capillary\ Red\ Layer.tiff
    ├── OMAG\ Bscans
    └── xzIntensity
```

> :warning: The function `cgan/utils.py/generate_inferred_images` assumes that OMAG directories are named `OMAG Bscans`.

Similarly, the `checkpoint_directory` might look like this.
The checkpoints should automatically be generated as training is run (see training instructions). 
Again, the actual image files are not shown for brevity.

```
training_checkpoints/
├── checkpoint
├── ckpt-25.data-00000-of-00002
├── ckpt-25.data-00001-of-00002
└── ckpt-25.index
```

The resulting output is stored in an `experiment-{timestamp}` directory.
```
experiment-2020-05-16-171703/
├── 2015-10-27___512_2048_Horizontal_Images58
├── 2015-10-27___512_2048_Horizontal_Images6
├── 2015-10-27___512_2048_Horizontal_Images67
├── 2015-10-27___512_2048_Horizontal_Images72
├── 2015-10-27___512_2048_Horizontal_Images73
├── README.md
├── discriminator_weights.h5
├── generator_weights.h5
└── logs
```

> :warning: Image generation - especially for large datasets - tends to take a very long time.
