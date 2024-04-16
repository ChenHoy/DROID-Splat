import shutil
from termcolor import colored
import ipdb
import os
import random

import numpy as np
import torch
import hydra

from src import config
from src.slam import SLAM
from src.datasets import get_dataset

"""
Run the SLAM system on a given dataset or on image folder.

You can configure the system using .yaml configs. See docs for reference
"""

def sys_print(msg: str) -> None:
    print(colored(msg, "white", "on_grey", attrs=["bold"]))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".",
        "..",
        ".git*",
        "*pycache*",
        "*build",
        "*.fuse*",
        "*_drive_*",
        "*pretrained*",
        "*output*",
        "*media*",
        "*.so",
        "*.pyc",
        "*.Python",
        "*.eggs*",
        "*.DS_Store*",
        "*.idea*",
        "*.pth",
        "*__pycache__*",
        "*.ply",
        "*exps*",
    )

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree(".", backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))


@hydra.main(version_base = None, config_path="./configs/", config_name="slam")
def run_slam(cfg):

    sys_print(f"\n\n** Running {cfg.data.input_folder} in {cfg.slam.mode} mode!!! **\n\n")

    setup_seed(43)
    torch.multiprocessing.set_start_method("spawn")

    # Save state for reproducibility
    backup_source_code(os.path.join(cfg.slam.output_folder, "code"))
    config.save_config(cfg, f"{cfg.slam.output_folder}/cfg.yaml")

    # Load dataset
    dataset = get_dataset(cfg, device=cfg.slam.device)

    # Run SLAM 
    slam = SLAM(cfg)
    slam.dataset = dataset
    slam.run(dataset)
    sys_print("Done!")



if __name__ == "__main__":
    run_slam()