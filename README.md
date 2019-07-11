# oct-opus
Image processing for OCT retina and cornea cross-sections.


## Setup Procedure (for ecelinux)

1. Go to eceubuntu4 via ```ssh username@eceubuntu4.uwaterloo.ca``` (may have to use ```username@ecelinux4.uwaterloo.ca``` as proxy).

2. ```cd /private/fydp1``` to access data and code.

3. Run ```source ./venv/bin/activate``` to initialize the Python environment.

4. Start Jupyter NoteBook: ```jupyter notebook```. This will startup a Jupyter server.

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

5. Go to another terminal and run ```ssh -NL 8000:localhost:8888 username@eceubuntu4.uwaterloo.ca``` on your own machine (not ecelinux) to setup the ssh tunnel.

    You may need to replace 8888 with the port number that you received. The prompt should hang.

6. Go to ```http://localhost:8000/?token=96e57ab83cd927178dd00e463bb4af11b54053d05829041a```, replacing the token in the URL with your own.

7. Use Ctrl-C to close the ssh tunnel and the Jupyter server.

## Other General Recommendations and Links

* Adding the following to your local machine's `~/.ssh/config` file will automatically do the port forwarding stuff every time you run the command `ssh workstation` (coupling together the SSHing in and also forwarding your 8888 port to the server's):
```
Host workstation
    HostName eceUbuntu4.uwaterloo.ca
    User <yourusername>
    LocalForward 8888 localhost:8888
    LocalForward 6006 localhost:6006
```
There are other things you can do, like setting up a `ProxyJump` via another campus server that happens to be accessible from outside, so that you don't have to launch a VPN or be physically present on campus to directly connect to `eceUbuntu4.uwaterloo.ca`.

* Use `gpustat` to keep an eye on GPU stats like temperature as you run training.

* Use `ls -hal` to keep an eye on groups and permissions for files in `/private/fydp1/oct-opus`, make sure group `fydp2019a` has access (Pei Lin has been using 770 perms so far with success), `chgrp` or `chmod` as necessary to ensure this.

* Once you're SSHed in and wanting to start your Jupyter Notebook server, first spin up `tmux` (`tmux new -s coolcats`). Then you can Ctrl+B, D to detach from that tmux session anytime (or lose your Internet connection and Jupyter will keep running in the background. Then `tmux attach -t coolcats` to re-attach to that tmux session. [More information here](https://towardsdatascience.com/jupyter-and-tensorboard-in-tmux-5e5d202a4fb6).

