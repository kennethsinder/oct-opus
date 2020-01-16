import glob
from compare import main as compare_main
import os
import os.path
import sys

if __name__ == '__main__':
    for run_path in glob.glob('./RUN_*'):
        print('-------------{}--------------'.format(run_path))
        for dataset_path in glob.glob(os.path.join(run_path, 'predicted-epoch-5', '20*')):
            dataset_name = os.path.basename(os.path.normpath(dataset_path))
            f_1 = os.path.join(sys.argv[1], 'all_data_enface', dataset_name, 'OMAG Bscans', 'multi_slice_sum.png')
            f_2 = os.path.join(dataset_path, 'multi_slice_sum.png')
            compare_main(f_1, f_2)

