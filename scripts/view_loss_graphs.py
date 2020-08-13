"""
Script for viewing smoothed loss graphs from the CSV output
produced by TensorBoard. Developed as an alternative to TensorBoard's graphs,
which show un-smoothed background spikes, when we only care to plot the smoothed
graph for each type of loss.

Some assumptions:
- CSV exports are for granular (every 200 training steps) losses for Fold #0.
- The 4 types of loss that the pix2pix cGAN code in this repo logs are the ones
exported in the 4 CSVs.
"""

from typing import List
import pandas as pd
import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def process_values(df, s_factor):
    values = df["Value"].to_list()
    df["Smooth"] = smooth(values, s_factor)
    return df


def plot_values(df, title):
    df.plot(x="Step", y=["Smooth"], legend=None)

    # grid lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    # metadata
    plt.title(title)
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    # load raw data
    disc_loss = pd.read_csv("run-.-tag-disc_loss_granular_fold_0.csv")
    gen_loss = pd.read_csv("run-.-tag-gen_gan_loss_granular_fold_0.csv")
    gen_l1_loss = pd.read_csv("run-.-tag-gen_l1_loss_granular_fold_0.csv")
    gen_total_loss = pd.read_csv("run-.-tag-gen_total_loss_granular_fold_0.csv")

    # smooth values
    smoothing_factor = 0.99
    disc_loss = process_values(disc_loss, smoothing_factor)
    gen_loss = process_values(gen_loss, smoothing_factor)
    gen_l1_loss = process_values(gen_l1_loss, smoothing_factor)
    gen_total_loss = process_values(gen_total_loss, smoothing_factor)

    # plot values
    plot_values(disc_loss, "disc_loss, SF={}".format(smoothing_factor))
    plot_values(gen_loss, "gen_gan_loss, SF={}".format(smoothing_factor))
    plot_values(gen_l1_loss, "gen_l1_loss, SF={}".format(smoothing_factor))
    plot_values(gen_total_loss, "gen_total_loss, SF={}".format(smoothing_factor))
