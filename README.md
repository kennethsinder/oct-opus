# oct-opus
Image processing for OCT retina and cornea cross-sections.


####Setup Procedure (for ecelinux)

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
