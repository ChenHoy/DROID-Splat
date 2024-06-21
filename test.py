import shutil
from termcolor import colored
import ipdb
import os
import random
import hydra

import numpy as np
import torch

from src.test.testbed import SlamTestbed as SLAM
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


def get_in_the_wild_heuristics(ht: int, wd: int, strategy: str = "generic") -> torch.Tensor:
    """We do not have camera intrinsics on in-the-wild data. In order for this to converge, we
    need a good initialize guess. There are two strategies to do this: i) generc ii) teeds from DeepV2D
    """
    if strategy == "generic":
        fx = fy = (wd + ht) / 2
        cx, cy = wd / 2, ht / 2
    else:
        fx = fy = wd * 1.2
        cx, cy = wd / 2, ht / 2
    return torch.Tensor([fx, fy, cx, cy], dtype=torch.float32)


@hydra.main(version_base=None, config_path="./configs/", config_name="slam")
def run_slam(cfg):

    output_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    setup_seed(43)
    torch.multiprocessing.set_start_method("spawn")

    sys_print("#######################################")
    sys_print("TESTBED")
    sys_print("#######################################")
    sys_print(f"\n\n** Running {cfg.data.input_folder} in {cfg.mode} mode!!! **\n\n")

    dataset = get_dataset(cfg, device=cfg.device)
    slam = SLAM(cfg, dataset=dataset, output_folder=output_folder)

    sys_print(f"Running on {len(dataset)} frames")
    slam.run(dataset)
    sys_print("Done!")


if __name__ == "__main__":
    run_slam()
