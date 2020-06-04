# Procedures

## Pre and Post Processing

### Flattening Script (Pre)

Cross-sections with curvature can be visually flattened
to produce a consistant final en-face image
and more standard looking cross-sections. We
have provided a modified version of
Han Le's (han.le@uwaterloo.ca) MATLAB script
which uses a quadratic approximation of the curvature
to perform the flattening.

(TODO: Additional info)

### Enface Generation (Post)

TODO (talk about `enface.py` and maybe
link to it - mention it works for both
the OMAG Bscans/xzIntensity folder structure
and the flat inferred folder structure of
an `experiment-<date/time>` directory.)

### Histogram Equalization (Post)

[Histogram equalization](https://en.wikipedia.org/wiki/Histogram_equalization) is a technique
used in image processing to normalize the pixel values of an image
by balancing the frequency of each pixel intensity (i.e. flattening
the histogram). It is particularly useful as a "post-processing"
step to run after generating predicted OCTA-like datasets using
our software.

The Python code for histogram equalization
is quite simple (with the use of a few dependencies
you can install with `pip install -r requirements.txt` using our provided requirements file in this repository).
The `scripts/post-normalizer.py` script is
a Python CLI with only one command-line argument: the path to an `experiment-...`
folder (or any other folder containing
subfolders which each contain a sequence
of numbered cross-sections) as would be
produced by using our software to generate
predictions for a test set.

It then performs histogram normalization of
every single image in-place (i.e. saving
the result and overwriting the previous
non-normalized image).

Alternatively, to only normalize en-face
(C-scan) images, there is an optional
`-n` command line flag that can be passed
to the enface generation script (`enface.py`)
described in the previous section. This will
leave the cross-sections untouched, but when
generating enfaces for each sequence of
cross-sections, it will perform a histogram
equalization step for the saved en-faces.

Remember that the en-face tool works for
both the `experiment-...` folder structure
(that is, for predicted images using the
trained pix2pix model) as well as raw
data folders (e.g. `all_data_original.tar.gz`).
See the "Data Set Conventions" section for
more information about these folder structures.

## Data Set Conventions

TODO

## Assumptions

TODO

