#! /usr/bin/env python3

import os
import ipdb
import argparse

import numpy as np
from PIL import Image

"""
Change the colors of a VOC style mask to the right color value of the rest of the totalrecon masks.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Convert json files to dense pixel masks")
    parser.add_argument("--voc_dir", type=str, help="Path to the directory containing the totalrecon style masks")
    return parser.parse_args()


def get_label_space(mask_dir: str) -> np.ndarray:
    # Sample the first image and check for values
    sample_file = os.listdir(mask_dir)[0]
    mask = np.array(Image.open(os.path.join(mask_dir, sample_file)))
    plot_img(mask)
    return np.unique(mask)


def plot_img(img):
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.axis("off")
    plt.show()


# humandog-human
def main(args):
    masks_dir = args.voc_dir
    target_dir = os.path.join(args.voc_dir, "../masks")
    # target_dir = os.path.join(args.voc_dir, "../test")

    # Red to Blue happens often for us
    # The annotator stored in range(label_space), which is usually [0, 10] for us
    # meanwhile TotalRecon stores label images in the range [0, 255] with full RGB color values
    # NOTE this script is not needed when we work with other datasets, which store in minimal range
    # label_map = {"source": [1], "target": [[0, 0, 128]]}
    label_map = {"source": [1, 8], "target": [[0, 0, 128], [254, 254, 254]]}
    # label_map = {"source": [8, 1], "target": [[0, 0, 128], [254, 254, 254]]}

    assert len(label_map["source"]) == len(label_map["target"]), "Source and target labels must be the same length"

    for src_file in os.listdir(masks_dir):
        if not src_file.endswith(".png") or src_file.endswith(".jpg") or src_file.endswith(".jpeg"):
            continue
        src_path = os.path.join(masks_dir, src_file)
        mask_src = Image.open(src_path)
        mask_src = np.array(mask_src)

        rgb_target = np.zeros(mask_src.shape + (3,))

        # Get which pixel we have to relabel
        new_label_masks = []
        for label in label_map["source"]:
            new_label_masks.append(mask_src == label)
        # Assign new label values
        # NOTE this assumes a correct ordering in the list in label_map["source"] and label_map["target"]
        for i, target_label in enumerate(label_map["target"]):
            rgb_target[new_label_masks[i]] = target_label

        # plot_img(rgb_target)
        mask_target = Image.fromarray(rgb_target.astype(np.uint8))
        mask_target.save(os.path.join(target_dir, src_file))


if __name__ == "__main__":
    args = parse_args()
    main(args)
