import shutil
from termcolor import colored
import ipdb
import os
import random

import numpy as np
import torch
import hydra

from src.slam import SLAM
from src.datasets import get_dataset

"""
Run the SLAM system on a given dataset or on image folder.
You can configure the system using .yaml configs. See docs for reference ...
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
        "*ext",
        "*thirdparty",
        "*.fuse*",
        "*_drive_*",
        "*pretrained*",
        "*output*",
        "*.png",
        "*.jpg",
        "*.jpeg",
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


@hydra.main(version_base=None, config_path="./configs/", config_name="slam")
def run_slam(cfg):

    output_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    setup_seed(43)
    torch.multiprocessing.set_start_method("spawn")
    # Save state for reproducibility
    backup_source_code(os.path.join(output_folder, "code"))

    sys_print(f"\n\n** Running {cfg.data.input_folder} in {cfg.mode} mode!!! **\n\n")
    dataset = get_dataset(cfg, device=cfg.device)
    slam = SLAM(cfg, output_folder=output_folder)

    sys_print(f"Running on {len(dataset)} frames")
    slam.run(dataset)
    sys_print("Done!")


if __name__ == "__main__":
    run_slam()
