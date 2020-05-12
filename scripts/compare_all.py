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

from scripts.compare import main as compare_main
import os
import os.path
import sys


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
    enfaces, of the shape (('multi_slice_sum.png), {'psnr_score': <some float>,
    ...}), ('multi_slice_max_norm.png', {'psnr_score': <some other float>, ...}).
    That is, the first element of each 2-tuple in the output will be a string
    (enface type filename),
    and the second element will be a dictionary of all scores for that enface type.
    Scores are calculated by *averaging* scores across each of the eyes under the
    given `experiment_dir`.
    """
    enface_scores = (('multi_slice_sum.png', {}), ('multi_slice_max_norm.png', {}))
    for enface_type in enface_scores:
        num_datasets = 0
        for dataset_path in [f.path for f in os.scandir(experiment_dir) if f.is_dir()]:
            dataset_name = os.path.basename(os.path.normpath(dataset_path))
            f_1 = os.path.join(data_dir, dataset_name, 'OMAG Bscans', enface_type[0])
            f_2 = os.path.join(dataset_path, enface_type[0])
            current_scores = compare_main(f_1, f_2)
            if not enface_type[1]:
                enface_type[1].update(current_scores)
            else:
                for score_type in current_scores:
                    enface_type[1][score_type] += current_scores.get(score_type, 0)
            num_datasets += 1
        for score_type in enface_type[1]:
            enface_type[1][score_type] /= float(num_datasets)
    return enface_scores


def main():
    experiment_dir = sys.argv[1]
    data_dir = sys.argv[2]
    enface_scores = score_for_test_results(experiment_dir, data_dir)

    # CSV Generation Code
    # First row is the string "Score Types" (since the 1st column will be the score types
    # for every subsequent row) and then the types of "enfaces" (sum and max norm),
    # and finally the column heading "Average", since the last column will have averages
    # for every row so we can get a summary value
    # for each type of scoring (e.g. SSIM) since each scoring type is one row.
    rows = [['Score Types'] + [enface_type[0] for enface_type in enface_scores] + ['Average']]
    for score_type in enface_scores[0][1]:
        row = [score_type]  # Add a string representing the type of score (e.g. "psnr_score")
        for enface_type in enface_scores:
            row.append(enface_type[1][score_type])
        row.append(sum(row[1:]) / len(row[1:]))  # Add the row average
        rows.append(row)

    import csv
    with open(os.path.join(experiment_dir, 'comparison.csv'), 'w') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    main()
