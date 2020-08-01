"""
Command-line script for comparing all of the predicted enfaces
in a given experiment directory to the enfaces created from
the real OMAGs in an objective way and generating a CSV
file in the experiment folder summarizing the scores.

Syntax: `python compare_all.py <experiment_dir> <data_dir>`

On Sharcnet, for convenience, this script should be launched
from the wrapper `sbatch compare_all_job.sh <experiment_dir>`,
which does the work of creating a job that extracts the ZIP
file `../all_data_flattened.tar.gz` into a temporary directory
and then feeding that location into this script as the
<data_dir>
"""

from compare import main as compare_main
import os
import os.path
import sys
import csv


def score_for_test_results(experiment_dir: str, data_dir: str):
    """ (str, str) -> tuple
    Takes in [1] a specific cGAN experiment directory,
    e.g. `./experiment-2020-05-11-170131`, containing one or more
    eyes' worth of predicted cross-sections under subfolders such as
    `2015-10-20___512_2048_Horizontal_Images21`, and
    [2] `data_dir` which is the path to all of the eyes' worth of
    B-ground truth OMAG data which we want to compare our experiment
    predictions to.

    Returns objective comparison metrics for both sum and max-norm
    enfaces in a 2-tuple of (enface type : str, data : dict), where
    data is keyed by the dataset name and the corresponding value is another
    dict with the objective scores (e.g. SSIM) for that specific enface.
    """
    enface_scores = (('multi_slice_sum.png', {}), ('multi_slice_max_norm.png', {}))
    for enface_type in enface_scores:
        for enface_path in os.listdir(experiment_dir):
            if enface_path.endswith(enface_type[0]):
                # <dataset_name>_<enface_type>
                dataset_name = enface_path[:-(len(enface_type[0])+1)]
            else:
                continue

            f_1 = os.path.join(data_dir, dataset_name, 'OMAG Bscans', enface_type[0])
            if not os.path.isfile(f_1):
                # Eye doesn't have a corresponding ground truth, nothing we can do.
                continue
            f_2 = os.path.join(experiment_dir, enface_path)

            current_scores = compare_main(f_1, f_2)
            enface_type[1].update({dataset_name: current_scores})
    return enface_scores


def main():
    experiment_dir = sys.argv[1]
    data_dir = sys.argv[2]
    enface_scores = score_for_test_results(experiment_dir, data_dir)

    # CSV Generation Code
    for enface_type in enface_scores:
        file_name = 'comparison_{}.csv'.format(enface_type[0].split('.')[0])
        rows = []
        sums = {}
        score_types = []
        for dataset_name in enface_type[1]:
            if not rows:
                score_types = [score_type for score_type in enface_type[1][dataset_name]]
                rows.append(['Dataset'] + score_types)

            row = [dataset_name]
            for score_type in score_types:
                score_value = enface_type[1][dataset_name][score_type]
                row.append(score_value)
                if score_type not in sums:
                    sums[score_type] = score_value
                else:
                    sums[score_type] += score_value
            rows.append(row)
        rows.append(['Average'] + [sums[score_type] / len(enface_type[1])
                                   for score_type in score_types])

        with open(os.path.join(experiment_dir, file_name), 'w') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)


if __name__ == '__main__':
    main()
