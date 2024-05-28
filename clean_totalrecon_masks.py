#! /usr/bin/env python3


"""
The background/foreground masks on TotalRecon are wrong for some frames. 
Every scene features these outliers and its likely due to the PointRender/Detectron2 preprocessing, 
which they used.

Normally, the background is stored as black values and foreground objects are tracked with a color value.
On outlier frames, the whole image is just white.
NOTE Turns out the images are not completely white, but there are very few (mostly just a single one) pixels with color (not visible to the human eye).
we simply detect these by assuming no image has a mean > 254.
"""

import os
import ipdb
from omegaconf import DictConfig

import torch
import cv2
import numpy as np

from src.datasets import get_dataset


def main(cfg: DictConfig):
    dataset = get_dataset(cfg)
    fpaths = dataset.mask_paths
    for frame, fpath in zip(dataset, fpaths):

        timestamp, image, depth, intrinsic, gt_pose, _ = frame
        mask = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if mask.mean() > 254:
            print(f"Detected outlier frame {fpath}")
            mask = np.zeros_like(mask)
            print(f"Overwriting with all zeros ...")
            cv2.imwrite(fpath, mask)


if __name__ == "__main__":
    scenes = [
        "cat0",
        "cat1",
        "cat2",
        "cat3",
        "dog0",
        "dog1",
        "human1",
        "human2",
        "humancat-animal",
        "humancat-human",
        "humandog-animal",
        "humandog-human",
    ]
    for scene in scenes:
        # Create a dummy config that works with our nested config system
        cfg = {
            "stride": 1,
            "mode": "mono",
            "with_dyn": True,
            "data": {
                "dataset": "totalrecon",
                "input_folder": "/media/data/totalrecon/",
                "scene": scene,
                "cam": {
                    "H": 960,
                    "W": 720,
                    "H_out": 480,
                    "W_out": 360,
                    "H_edge": 0,
                    "W_edge": 0,
                    "fx": 794.65,
                    "fy": 794.65,
                    "cx": 353.53,
                    "cy": 468.01,
                    "camera_model": "pinhole",
                },
            },
        }
        cfg = DictConfig(cfg)
        main(cfg)
