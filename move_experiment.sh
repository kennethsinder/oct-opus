# Supply run # as the only argument (for some run # that does not exist yet).
# e.g. ./move_experiment.sh 16

mkdir -p RUN_$1
mv predicted-* logs training_checkpoints comparison* slurm-* *.h5 RUN_$1/ 2>/dev/null

