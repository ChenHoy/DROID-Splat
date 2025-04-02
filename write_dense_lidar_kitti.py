import shutil
from termcolor import colored
from tqdm import tqdm
import ipdb
import yaml
import os
import random
import hydra
import logging
from omegaconf import OmegaConf

import numpy as np
import matplotlib.pyplot as plt

from src.datasets import get_dataset


@hydra.main(version_base=None, config_path="./configs/", config_name="slam")
def write_depth(cfg):

    output_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Save the cfg to yaml file
    with open(os.path.join(output_folder, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(cfg), f, default_flow_style=False)

    cfg.stride = 1  # Overwrite to every frame in any case
    cfg.mode = "rgbd"
    dataset = get_dataset(cfg, device=cfg.device)
    new_gt_path = os.path.join(dataset.input_folder, "depth", "lidar")
    os.makedirs(new_gt_path, exist_ok=True)

    # Main Loop which drives the whole system
    i = 0
    for frame in tqdm(dataset):
        if cfg.with_dyn and dataset.has_dyn_masks:
            timestamp, image, depth, intrinsic, gt_pose, static_mask = frame
        else:
            timestamp, image, depth, intrinsic, gt_pose = frame
            static_mask = None

        depth = depth.squeeze().cpu().numpy()

        # fig = plt.figure()
        # plt.imshow(depth)
        # plt.axis("off")
        # plt.show()

        # fig = plt.figure()
        # plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
        # plt.axis("off")
        # plt.show()

        fname = os.path.join(new_gt_path, f"{str(i).zfill(6)}.npy")
        np.save(fname, depth)

        i += 1


if __name__ == "__main__":
    write_depth()
