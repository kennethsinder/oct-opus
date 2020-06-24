# Sharcnet

Compute Canada's Sharcnet is a powerful
set of computing resources useful for
training the pix2pix model used in this software
as well as other image processing tasks involving the GPU.

A lot of useful scaffolding has been provided in this repository to make it easy to interact with Sharcnet
and the Slurm job manager in general. First, take a look at Compute Canada's documentation, which will be handy
to keep open as you work with Sharcnet: https://docs.computecanada.ca/wiki/Running_jobs

We used Sharcnet primarily for training and also running somewhat GPU-intensive pre- and post-processing
scripts. For generated inferred images, a powerful machine is not required and this can be done locally using
our graphical user interface (see the `gui.md` docs.)

The `sbatch` command is used to launch Sharcnet jobs.

- For all of the scripts that we created a Sharcnet wrapper for, these will end in `_job.sh` -- do not run these directly, but run them through `sbatch`.
- To run training, go to the home directory of the repository, wherever you `git clone`d or extracted it, and run `sbatch run_job.sh 3 -k 5`. The command-line flags are described in the `training_and_testing.md` doc, but in this example, it's setting up the training to run for 3 epochs per fold and K = 5
[folds cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation). Output logs
for the Sharcnet job will go into a `slurm-<job id>.out` file and otherwise running training is as we documented --
training checkpoints and predictions go to an `experiment-<date/time>/` folder
- After a k-folds run, or a non-k-folds prediction (testing mode) run of the above, you can generate a CSV file of objective comparison metrics describing the quality of your results using `sbatch ./scripts/compare_all_job.sh ./experiment-<date/time>`.
- Interactive jobs are [possible](https://docs.computecanada.ca/wiki/Running_jobs#Interactive_jobs) if you need access to a node with a GPU for up to 3 hours and actually issue commands on the command line directly to the machine instead of running the job in the background. It is recommended to use interactive jobs for debugging issues and figuring out input and output folders, and then switching to regular jobs once you know the code and command-line arguments are working.

:warning: Remember to change all occurrences of `def-vengu` in the `*_job.sh` scripts to reflect the actual Compute Canada account owner, and all references to `s2saberi` to reflect your personal username on the Graham login machines.

For convenience using the Graham machines, adding the following aliases to your `~/.bash_profile` is recommended:
```bash
# User specific environment and startup programs
alias gohome='cd ~/projects/def-vengu/<YOUR USERNAME>'
alias gogpu='salloc --time=3:0:0 --nodes=1 --ntasks-per-node=32 --mem=127000M --gres=gpu:2 --account=def-<YOUR PROJECT OWNER>'
```

- `gohome` will take you to your home directory, where you should have (1) all `*.tar.gz` files for raw data that will be extracted by the Sharcnet job before it runs training on that data and (2) the `oct-opus` repository extracted. That way,
you can do `gohome && cd oct-opus` and start GPU jobs from in there.
- `gogpu` will quickly give you an interactive GPU node for 3 hours so you can run any non-trivial CPU or GPU scripts or commands, like moving around and extracting or creating `.tar.gz` files or verifying if your code is working. Note that GPU nodes on Compute Canada never have Internet access, so you cannot run any GitHub commands from within a GPU node or upload logs to an online service. You can relinquish the interactive job using EOF (Ctrl+D) or the `exit` command.

:warning: Never run any CPU commands that take more than a small amount of time and memory on an SSH (login) node - default to requesting a GPU node when you need one.
