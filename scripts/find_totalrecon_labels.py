#! /usr/bin/env python3

import os
import ipdb
import argparse

import numpy as np
from PIL import Image

"""
We can manually annotate any image/video we want using an annotation pipeline based on SAM. 
Sometimes we forgot to use the correct label space, so we need to find out the label space.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Convert json files to dense pixel masks")
    parser.add_argument("--masks_dir", type=str, help="Path to the directory containing the totalrecon style masks")
    return parser.parse_args()


def get_label_space(mask_dir: str) -> np.ndarray:
    # Sample the first image and check for values
    all_files = sorted(os.listdir(mask_dir))
    sample_file = all_files[0]
    mask = np.array(Image.open(os.path.join(mask_dir, sample_file)))
    plot_img(mask)
    ipdb.set_trace()
    return np.unique(mask)


def plot_img(img):
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.axis("off")
    plt.show()


def main(args):
    masks_dir = args.masks_dir
    label_space = get_label_space(masks_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
