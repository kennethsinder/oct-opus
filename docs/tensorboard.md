# TensorBoard

When running training, logs in the [TensorBoard](https://www.tensorflow.org/tensorboard) format are saved to the `experiment-<date/time>/logs` directory. These logs are useful for observing what's happening if you make changes to the neural
net architecture or code.
Both scalar values representing the different types of loss values over the course of training, and images of
side-by-side cross-sections, are logged. To view the logs:

1. Download the `logs` folder onto a local computer with a GUI, using the command-line tool `scp` or any other method.
1. Make sure Python 3 (preferably Python 3.6.9) and `tensorboard` are installed using `pip install tensorboard`.
1. Run `tensorboard --logdir logs` from the command-line.
1. Go to the website that `tensorboard` spins up - it will probably point you to http://localhost:6006/
1. Observe the graphs of the losses. Our paper has a section describing what the different loss graphs should look like
over the course of training, and you can find similar explanatory documentation at Google's official [pix2pix tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_generator).

Note that for scalar loss value graphs, we log separate graphs for every fold in the case of a K-folds cross-validation
training run. You can just pick any of the 5 folds and analyze that graph without loss of generality unless one of them
looks anomalous.

Also note that we log two types of loss values: the ones ending in `_granular` are more granular, being
logged every 200 training steps, while the other graphs only show one value per epoch. Looking at the granular graphs is
recommended because a lot happens within one epoch that is lost when you look at the more coarse-grained values.

Also recommended is to apply smoothing in the left hand side of the TensorBoard user interface. A higher smoothing value
will give you a rolling average to show you the overall trend of each graph while showing the original graph as a fainter
trace in the background.
