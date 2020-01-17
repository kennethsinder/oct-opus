import glob
from compare import main as compare_main
import os
import os.path
import sys
import re

if __name__ == '__main__':
    for run_path in glob.glob('./RUN_*'):
        run_num = int(re.search(r'RUN_(\d+)', run_path).group(1))
        num_datasets = 0
        scores = None
        print('-------------{}--------------'.format(run_path))
        for dataset_path in glob.glob(os.path.join(run_path, 'predicted-epoch-5', '20*')):
            dataset_name = os.path.basename(os.path.normpath(dataset_path))
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
            print('Average {} for RUN_{} = {}'.format(score_type, run_num, scores[score_type] / num_datasets))

