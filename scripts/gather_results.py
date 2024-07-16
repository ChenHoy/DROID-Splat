#! /usr/bin/env python3

import os
from pathlib import Path
import argparse
from termcolor import colored
import pandas as pd
import json

"""
Gather statistics across multiple runs or sequences (e.g. of a dataset). This will simply 
read out the results for a couple of subfolders inside a single folder. Make sure 
you arranged all the different runs or sequences already in a folder structure.
"""

# Define the metrics keys you're interested in
rendering_metrics_keys = ["psnr", "ssim", "lpips", "l1_depth"]
tracking_metrics_keys = ["rmse", "mean", "median", "std"]


def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze experiment data")
    parser.add_argument("--data", "-i", type=str, help="Path to the folder containing experiment runs")
    return parser.parse_args()


def read_json(json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def main_print(msg: str, color: str = "white") -> None:
    print(colored(msg, color, "on_grey", attrs=["bold"]))


def metric_print(msg: str, color: str = "green") -> None:
    print(colored(msg, color, attrs=["bold"]))


def main(args):
    """Main function to collect and analyze data."""
    data_path = Path(args.data)

    # Initialize an empty dictionary to store statistics
    statistics = {"tracking_all": {}, "tracking_kf": {}, "rendering_kf": {}, "rendering_nkf": {}}
    for metric in tracking_metrics_keys:
        for key in statistics.keys():
            statistics[key][metric] = []
    for metric in rendering_metrics_keys:
        for key in statistics.keys():
            statistics[key][metric] = []

    # Go through each experiment subfolder
    for subfolder in data_path.iterdir():
        eval_dir = subfolder / "evaluation"
        tracking_dir = eval_dir / "odometry"
        tracking_all = read_json(tracking_dir / "all" / "stats_final.json")
        tracking_kf = read_json(tracking_dir / "keyframes" / "stats_final.json")
        for key in tracking_metrics_keys:
            if key in tracking_all:
                statistics["tracking_all"][key].append(tracking_all[key])
            if key in tracking_kf:
                statistics["tracking_kf"][key].append(tracking_kf[key])

        rendering_dir = eval_dir / "rendering"
        # We dont always render when we e.g. just run the Tracker
        if rendering_dir.exists():
            rendering_metrics = pd.read_csv(rendering_dir / "evaluation_results.csv")
            rendering_kf = rendering_metrics[rendering_metrics["eval_on_keyframes"] == True]
            rendering_nkf = rendering_metrics[rendering_metrics["eval_on_keyframes"] == False]
            for key in rendering_metrics_keys:
                if key in rendering_kf.columns:
                    statistics["rendering_kf"][key].append(rendering_kf[key].values[0])
                if key in rendering_nkf.columns:
                    statistics["rendering_nkf"][key].append(rendering_nkf[key].values[0])

    # Calculate mean, median, std for each metric
    main_print(f"Evaluation statistics for: {data_path}\n")
    for eval_kind in statistics.keys():

        main_print(f"\n {eval_kind} \n", "red")
        for key in statistics[eval_kind].keys():
            if len(statistics[eval_kind][key]) == 0:
                continue
            metric_print(f"{key}", "green")
            print(f"Mean: {pd.Series(statistics[eval_kind][key]).mean()}")
            print(f"Median: {pd.Series(statistics[eval_kind][key]).median()}")
            print(f"Std: {pd.Series(statistics[eval_kind][key]).std()}")
            print("\n")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
