# oct-opus

Image processing for OCT retina and cornea cross-sections.

## Run Training

1. Activate the virtual environment.

2. Move all unrelated checkpoints out of the `./training_checkpoints` folder.

3. Execute `./run.sh <STARTING_EPOCH> <ENDING_EPOCH>`. The bash script will run `python run.py` with the appropriate parameters.

If you are training a model from scratch, `<STARTING_EPOCH>` should be 1. If you are restoring a model from a checkpoint, `<STARTING_EPOCH>` should be the next epoch to train on (eg. if the model has already done 2 epochs of training, `<STARTING_EPOCH>` should be 3).

This is a stupid procedure, because we're actually spawning a new `run.py` program for every epoch, and then `run.py` loads a checkpoint (if the `--restore` flag is set, that is) and then runs just one epoch of training, saves a checkpoint, and bails out. And `run.sh` is the harness that keeps spawning new `run.py` programs for each epoch. The reason we're doing this dumb thing is because of a memory leak that we don't have time to look into right now - the main thing is just to evaluate the pix2pix baseline, and the results should be the same with this approach, and the memory consumption will be under control so we can run large numbers of epochs.

The only thing we have to watch out for is SSD space (TODO: modify the code to only keep training checkpoints in a certain window and auto-delete old ones whenever we save a new one. This code might live at the bottom of `train.py` - take a look.)

## Generate Inferred Images

To generate sets of predicted images based on B-scans within test sets, put all relevant test sets within the `/private/fydp1/testing-data` folder. Note that `utils.py` will assume 4 acquisitions (i.e. 4 B-scans) for each particular cross section, so 4:1 ratio between OMAGs:Bscans. Then, run `python run.py predict`. It will create a `predicted` folder containing subfolders for every eye in the test set and in there will be the inferred images, of the same cardinality as the number of OMAGs (so you can compare each inferred image in sequence to the OMAG to see how good of a job it did at enhancing capillaries).

## View Tensorboard

1. SSH into `eceubuntu4` using `ssh workstation`.

2. Activate the virtual environment.

3. Execute `tensorboard --logdir logs/ --samples_per_plugin images=100`.

4. Open `localhost:6006` in a web browser.

You should see something like this:

![Screenshot of the Tensorboard UI](tensorboard_screen.png)

The x-axis is measured by training step.

You can view the Tensorboard dashboard while the model is training, or after it's done. Since the model is trained by executing `run.py` for each epoch, each epoch has its own line in the graphs. In the future, hopefully all the epochs will log to the same, continuous line.

Tensorboard logs will be stored in `./logs/`. eg. `./logs/31-07-2019_10:32:53/` contains the logs for training that started on 31-07-2019 at 10:32:53. For every execution of `run.sh`, a new folder will be created in './logs'.

Within `./logs/31-07-2019_10:32:53/`, there will be folders containing the data for each epoch, eg. `./logs/31-07-2019_10:32:53/epoch_1`.

Feel free to rename, move, or delete folders; Tensorboard will update accordingly.

## Generate Enface Images

1. Remember to ssh using the `-Y` flag

2. Execute `python3 ./plot.py` and follow prompts

3. Change `LOW_BOUND_LAYER` and `HIGH_BOUND_LAYER` in `plot.py` as necessary

4. Save image using the matplotlib UI

## General Recommendations and Links

- Adding the following to your local machine's `~/.ssh/config` file will automatically do the port forwarding stuff every time you run the command `ssh workstation` (coupling together the SSHing in and also forwarding your 8888 port to the server's):

```
Host workstation
    HostName eceUbuntu4.uwaterloo.ca
    User <yourusername>
    LocalForward 8888 localhost:8888
    LocalForward 6006 localhost:6006
```

There are other things you can do, like setting up a `ProxyJump` via another campus server that happens to be accessible from outside, so that you don't have to launch a VPN or be physically present on campus to directly connect to `eceUbuntu4.uwaterloo.ca`.

- Use `gpustat` to keep an eye on GPU stats like temperature as you run training.

- Use `ls -hal` to keep an eye on groups and permissions for files in `/private/fydp1/oct-opus`, make sure group `fydp2019a` has access (Pei Lin has been using 770 perms so far with success), `chgrp` or `chmod` as necessary to ensure this.

- Once you're SSHed in and wanting to start your Jupyter Notebook server, first spin up `tmux` (`tmux new -s coolcats`). Then you can Ctrl+B, D to detach from that tmux session anytime (or lose your Internet connection and Jupyter will keep running in the background. Then `tmux attach -t coolcats` to re-attach to that tmux session. [More information here](https://towardsdatascience.com/jupyter-and-tensorboard-in-tmux-5e5d202a4fb6).


## Run Jupyter Notebook (Deprecated)

1. Go to eceubuntu4 via `ssh username@eceubuntu4.uwaterloo.ca` (may have to use `username@ecelinux4.uwaterloo.ca` as proxy).

2. `cd /private/fydp1` to access data and code.

3. Run `source ./venv/bin/activate` to initialize the Python environment.

4. Start Jupyter NoteBook: `jupyter notebook`. This will startup a Jupyter server.

   You should see something similar to the following.

   ```bash
   (venv) pl3li@eceubuntu4: /private/fydp1:$ jupyter notebook
   [I 16:24:34.605 NotebookApp] Serving notebooks from local directory: /private/fydp1
   [I 16:24:34.605 NotebookApp] The Jupyter Notebook is running at:
   [I 16:24:34.605 NotebookApp] http://localhost:8888/?token=96e57ab83cd927178dd00e463bb4af11b54053d05829041a
   [I 16:24:34.605 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
   [W 16:24:34.630 NotebookApp] No web browser found: could not locate runnable browser.
   [C 16:24:34.630 NotebookApp]

       To access the notebook, open this file in a browser:
           file:///home/pl3li/.local/share/jupyter/runtime/nbserver-28660-open.html
       Or copy and paste one of these URLs:
           http://localhost:8888/?token=96e57ab83cd927178dd00e463bb4af11b54053d05829041a
   ```

   Make note of the port number that is provided. You may be assigned a port number that is different from 8888.

5. Go to another terminal and run `ssh -NL 8000:localhost:8888 username@eceubuntu4.uwaterloo.ca` on your own machine (not ecelinux) to setup the ssh tunnel.

   You may need to replace 8888 with the port number that you received. The prompt should hang.

6. Go to `http://localhost:8000/?token=96e57ab83cd927178dd00e463bb4af11b54053d05829041a`, replacing the token in the URL with your own.

7. Use Ctrl-C to close the ssh tunnel and the Jupyter server.
