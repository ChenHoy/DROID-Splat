import argparse
import shutil
import ipdb
import os
import random

import numpy as np
import torch

from src import config
from src.slam import SLAM
from src.datasets import get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_frames", type=int, default=-1, help="Only [0, max_frames] Frames will be run")
    parser.add_argument("--only_tracking", action="store_true", help="Only tracking is triggered")
    parser.add_argument("--make_video", action="store_true", help="to generate video as in our project page")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="input folder, this have higher priority, can overwrite the one in config file",
    )
    parser.add_argument(
        "--output", type=str, help="output folder, this have higher priority, can overwrite the one in config file"
    )
    parser.add_argument(
        "--mode", type=str, help="slam mode: mono, prgbd, rgbd or stereo", choices=["mono", "prgbd", "rgbd", "stereo"]
    )
    parser.add_argument(
        "--image_size",
        nargs="+",
        default=None,
        help="image height and width, this have higher priority, can overwrite the one in config file",
    )
    parser.add_argument("--stride", type=int, default=None, help="stride for frame sampling")
    parser.add_argument("--opt_intr", action="store_true", help="optimize intrinsics in bundle adjustment as well")
    parser.add_argument(
        "--camera_model",
        type=str,
        default="pinhole",
        choices=["pinhole", "mei"],
        help="camera model used for projection",
    )
    parser.add_argument(
        "--calibration_txt",
        type=str,
        default=None,
        help="calibration parameters: fx, fy, cx, cy, this have higher priority, can overwrite the one in config file",
    )
    parser.add_argument("--evaluate", type=bool, default=False, help="Enter evaluation mode. Deactivate gui and visualization.")
    return parser.parse_args()


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


def set_args(args, cfg):
    if args.mode is not None:
        cfg["mode"] = args.mode
    if args.only_tracking:
        cfg["only_tracking"] = True
    if args.image_size is not None:
        cfg["cam"]["H_out"], cfg["cam"]["W_out"] = args.image_size
    if args.calibration_txt is not None:
        cfg["cam"]["fx"], cfg["cam"]["fy"], cfg["cam"]["cx"], cfg["cam"]["cy"] = np.loadtxt(
            args.calibration_txt
        ).tolist()

    cfg['evaluate'] = args.evaluate
    
    assert cfg["mode"] in ["rgbd", "prgbd", "mono", "stereo"], "Unknown mode: {}".format(cfg["mode"])
    cfg["stride"] = args.stride if args.stride is not None else cfg["stride"]

    # Overwrite directory from cli manually
    cfg["data"]["input_folder"] = args.input_folder if args.input_folder is not None else cfg["data"]["input_folder"]
    if args.output is None:
        output_dir = cfg["data"]["output"]
    else:
        output_dir = args.output
        cfg["data"]["output"] = output_dir
    return output_dir, cfg


def typecheck_cfg(cfg):
    floats = ["fx", "fy", "cx", "cy", "png_depth_scale"]
    ints = ["H", "W", "H_out", "W_out", "H_edge", "W_edge"]
    for k, v in cfg["cam"].items():
        if k == "calibration_txt":
            continue
        if k in floats:
            cfg["cam"][k] = float(v)
        elif k in ints:
            cfg["cam"][k] = int(v)
        else:
            raise ValueError(f"Unknown type {type(k)} for '{k}'. This should be either float or int")
    return cfg


if __name__ == "__main__":
    setup_seed(43)
    args = parse_args()

    torch.multiprocessing.set_start_method("spawn")

    cfg = config.load_config(args.config, "./configs/go_gaussian_slam.yaml")
    output_dir, cfg = set_args(args, cfg)
    cfg = typecheck_cfg(cfg)

    print(f"\n\n** Running {cfg['data']['input_folder']} in {cfg['mode']} mode!!! **\n\n")
    print(args)
    # Save state for reproducibility
    # backup_source_code(os.path.join(output_dir, "code"))
    # config.save_config(cfg, f"{output_dir}/cfg.yaml")

    # Run SLAM
    ## index, color_data, depth_data, intrinsic, pose
    ## color data is RGB image, depth_data is WxH depth image
    dataset = get_dataset(cfg, args, device=args.device)

    # print(dataset)
    # index, color_data, depth_data, intrinsic, pose = dataset[5]
    # print(color_data.shape,depth_data.shape)
    slam = SLAM(args, cfg)
    slam.set_dataset(dataset)
    slam.run(dataset)
    slam.terminate(rank=-1, stream=dataset)

    print("Done!")
