#! /usr/bin/env python3

from typing import List
import glob
import os
import ipdb

import numpy as np


def load_poses(path, n_img: int) -> np.ndarray:
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for i in range(n_img):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        w2c = np.linalg.inv(c2w)
        poses.append(c2w)
    return poses


def write_out_kitti_style(traj: List[np.ndarray], outfile: str = "test.txt") -> None:
    """Given a list of 4x4 homogeneous matrices, write out the poses in KITTI style format.
    For each pose, we write a line in a .txt file as follows:
        a b c d
        e f g h -> a b c d e f g h i j k l
        i j k l
        0 0 0 1
    """
    with open(outfile, "w") as f:
        for pose in traj:
            pose = pose.flatten()
            for i in range(12):
                if i == 11:
                    f.write(str(pose[i]))  # Dont leave a trailing space
                else:
                    f.write(str(pose[i]) + " ")
            f.write("\n")


def main():
    input_folder = "/media/data/Replica/office0/"
    pose_path = os.path.join(input_folder, "traj.txt")
    color_paths = sorted(glob.glob(os.path.join(input_folder, "results/frame*.jpg")))
    n_img = len(color_paths)
    poses = load_poses(pose_path, n_img)
    ipdb.set_trace()
    write_out_kitti_style(poses, "test.txt")


if __name__ == "__main__":
    main()
