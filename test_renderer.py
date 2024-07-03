import torch
import hydra
import torch
import torch.multiprocessing as mp
import numpy as np
import cv2
import time
import open3d as o3d
import lietorch

from typing import List, Dict, Optional


from src.gaussian_splatting.scene.dynamic_gaussian_model import DynamicGaussianModel
from src.gaussian_splatting.scene.gaussian_model import GaussianModel
from src.gaussian_splatting.gaussian_renderer import render, render_cg

from src.datasets import get_dataset
from src.gaussian_splatting.camera_utils import Camera
from src.utils.multiprocessing_utils import clone_obj
from src.gaussian_splatting.utils.graphics_utils import (
    getProjectionMatrix2,
    getWorld2View2,
    focal2fov,
)



def camera_from_frame(
    idx: int,
    image: torch.Tensor,
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_init: Optional[torch.Tensor] = None,
    depth: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    device="cuda",
    ):

    fx, fy, cx, cy = intrinsics
    z_near = 0.001
    z_far = 10000.0

    height, width = image.shape[-2:]
    fovx, fovy = focal2fov(fx, width), focal2fov(fy, height)
    projection_matrix = getProjectionMatrix2(z_near, z_far, cx, cy, fx, fy, width, height)
    projection_matrix = projection_matrix.transpose(0, 1).to(device=device)

    return Camera(
        idx,
        image.contiguous(),
        depth_init,
        depth,
        w2c,
        projection_matrix,
        (fx, fy, cx, cy),
        (fovx, fovy),
        (height, width),
        device=device,
        mask=mask,
    )

def get_pose_optimizer(frames: List):
    """Creates an optimizer for the camera poses for all provided frames."""
    opt_params = []
    for cam in frames:
        opt_params.append(
            {"params": [cam.cam_rot_delta], "lr": 0.0003, "name": "rot_{}".format(cam.uid)}
        )
        opt_params.append(
            {
                "params": [cam.cam_trans_delta],
                "lr": 0.0001,
                "name": "trans_{}".format(cam.uid),
            }
        )
        opt_params.append({"params": [cam.exposure_a], "lr": 0.01, "name": "exposure_a_{}".format(cam.uid)})
        opt_params.append({"params": [cam.exposure_b], "lr": 0.01, "name": "exposure_b_{}".format(cam.uid)})

    return torch.optim.Adam(opt_params)
    

@hydra.main(version_base=None, config_path="./configs/", config_name="slam")
def main(cfg):
    dataset = get_dataset(cfg, device=cfg.device)
    torch.multiprocessing.set_start_method("spawn")
    kf_mng_params = cfg.mapping.keyframes.mapping
    sh_degree = 3 if cfg.mapping.use_spherical_harmonics else 0

    gaussians = GaussianModel(sh_degree, config=cfg.mapping.input)

    gaussians.init_lr(cfg.mapping.opt_params.init_lr)
    gaussians.training_setup(cfg.mapping.opt_params)

    q_main2vis = mp.Queue()
    gui_done = torch.zeros((1)).int().share_memory_()

    bg_color = [1, 1, 1]  # White background
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    z_near = 0.001
    z_far = 100000.0

    timestamp, image, depth, intrinsic, gt_pose = dataset[0]
    image = image.squeeze(0).to(cfg.device)
    intrinsic = intrinsic.to(cfg.device)
    gt_pose = gt_pose.squeeze().to(cfg.device)

    if depth is not None:
        depth = depth.to(cfg.device)


    w2c = lietorch.SE3(gt_pose).inv().matrix()

    cam = camera_from_frame(timestamp, image, w2c, intrinsic, depth=depth, depth_init=depth)
    cam.update_RT(cam.R_gt, cam.T_gt)

    pose_opt = get_pose_optimizer([cam])

    gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)
    gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)
    gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)

    print("Gaussians: ", len(gaussians))

    if cfg.render == "cg":
        render_fct = render_cg
    else:
        render_fct = render

    iters = 1000
    t0 = time.time()
    for i in range(iters):
        print(f"Iteration: {i+1}/{iters}", end="\r")
        render_pkg = render_fct(cam, gaussians, cfg.mapping.pipeline_params, background, device=cfg.device)

        if cfg.render == "cg":
            image = render_pkg[0]
        else:
            image = render_pkg["render"]

        loss = torch.abs(image - cam.original_image).mean()
        loss.backward()

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad()
            pose_opt.step()
            pose_opt.zero_grad()

    t1 = time.time()
    print("Loss", loss.item())
    print(f"{(t1 - t0)/iters}s per iteration")


    rgb = np.uint8(255 * image.detach().cpu().numpy().transpose(1, 2, 0))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        #cv2.imshow("Render", bgr)
    cv2.imwrite(f"/home/leon/test_{cfg.render}.png", bgr)


    print("Done")


if __name__ == "__main__":
    main()