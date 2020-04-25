import glob
from scripts.compare import main as compare_main
import os
import os.path
import sys
import re


def score_for_test_results(results_folder, run_num):
    print('-------------{}--------------'.format(results_folder))
    num_datasets = 0
    scores = None
    for dataset_path in glob.glob(os.path.join(results_folder, '20*')):
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        print('-------------{}--------------'.format(dataset_name))
        f_1 = os.path.join(sys.argv[1], 'all_data_enfaces', dataset_name, 'OMAG Bscans', 'multi_slice_sum.png')
        f_2 = os.path.join(dataset_path, 'multi_slice_sum.png')
        current_scores = compare_main(f_1, f_2)
        if not scores:
            scores = current_scores
        else:
            for score_type in current_scores:
                scores[score_type] += current_scores.get(score_type, 0)
        num_datasets += 1
    for score_type in scores:
        scores[score_type] /= float(num_datasets)
        print('Average {} for RUN_{} = {}'.format(score_type, run_num, scores[score_type]))
    return scores


if __name__ == '__main__':
    print([x for x in sys.argv])
    k_folds_mode = len(sys.argv) == 3
    run_paths = './{}'.format(sys.argv[2]) if k_folds_mode else './RUN_*'
    for run_path in glob.glob(run_paths):
        run_num = int(re.search(r'RUN_(\d+)', run_path).group(1))
        print('-------------{}--------------'.format(run_path))
        epoch_nums_glob = 'predicted-epoch-*' if k_folds_mode else 'predicted-epoch-5'
        all_epoch_scores = {}
        for results_folder in glob.glob(os.path.join(run_path, epoch_nums_glob)):
            if not k_folds_mode and (
                    'predicted-epoch-5' not in results_folder or 'predicted-epoch-50' in results_folder):
                continue
            epoch_num = int(re.search(r'epoch-(\d+)', results_folder).group(1))
            if k_folds_mode:
                all_epoch_scores[epoch_num] = score_for_test_results(results_folder, run_num)

        if k_folds_mode:
            # First row is the string "Score Types" (since the 1st column will be the score types
            # for every subsequent row) and then the fold numbers in order, and finally the column heading
            # "Average", since the last column will have averages for every row so we can get a summary value
            # for each type of scoring (e.g. SSIM) since each scoring type is one row.
            rows = [['Score Types'] + list(str(int(x / 5)) for x in sorted(all_epoch_scores.keys())) + ['Average']]
            for score_type in all_epoch_scores[list(all_epoch_scores.keys())[0]]:
                row = [score_type]  # Add a string representing the type of score (e.g. "psnr_score")
                for epoch_num in sorted(all_epoch_scores.keys()):
                    row.append(all_epoch_scores[epoch_num][score_type])     # Add fold-by-fold values to the row
                row.append(sum(row[1:]) / len(row[1:]))     # Add the row average
                rows.append(row)

            import csv
            with open('comparison.csv', 'w') as f:
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row)
