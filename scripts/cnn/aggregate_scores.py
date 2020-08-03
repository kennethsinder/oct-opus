import argparse
import csv
from os.path import join

HEADER = ['Dataset', 'psnr_score', 'mse_score', 'nrmse_score', 'ssim_score', 'mae_score']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root-dir', required=True)
    args = parser.parse_args()

    def aggregate(csv_file):
        aggregated = []
        for k in range(5):
            enface_dir = join(args.root_dir, str(k), 'enfaces/epoch_15')
            with open(join(enface_dir, csv_file), 'r') as file:
                reader = csv.reader(file)
                next(reader) # skip header row
                for row in reader:
                    # skip 'Average' row
                    if row[0] == 'Average':
                        continue
                    # convert strings into floats
                    new_row = []
                    new_row.append(row[0])
                    for score in row[1:]:
                        new_row.append(float(score))
                    aggregated.append(new_row)

        # calculate averages
        averages = ['Average', 0.0, 0.0, 0.0, 0.0, 0.0]
        for a in aggregated:
            for idx in range(1, 6):
                averages[idx] += a[idx]
        for idx in range(1, 6):
            averages[idx] /= len(aggregated)

        aggregated_file = join(args.root_dir, csv_file)
        print('Saving to {}'.format(aggregated_file))
        with open(aggregated_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(HEADER)
            for a in aggregated:
                writer.writerow(a)
            writer.writerow(averages)

    aggregate('comparison_multi_slice_max_norm.csv')
    aggregate('comparison_multi_slice_sum.csv')


if __name__ == '__main__':
    main()
