import torch
import hydra
import torch
import torch.multiprocessing as mp
import numpy as np
import cv2
import time
import open3d as o3d

from typing import List, Dict, Optional


from src.gaussian_splatting.scene.dynamic_gaussian_model import DynamicGaussianModel
from src.gaussian_splatting.scene.gaussian_model import GaussianModel
from src.gaussian_splatting.gaussian_renderer import render, render_dynamic

from src.datasets import get_dataset
from src.gaussian_splatting.gui import gui_utils, slam_gui
from src.gaussian_splatting.camera_utils import Camera
from src.gaussian_splatting.multiprocessing_utils import clone_obj
from src.gaussian_splatting.utils.graphics_utils import (
    getProjectionMatrix2,
    getWorld2View2,
    focal2fov,
)



def camera_from_frame(
    idx: int,
    image: torch.Tensor,
    depth: Optional[torch.Tensor],
    intrinsic: torch.Tensor,
    gt_pose: torch.Tensor,
    dynamic_mask: Optional[torch.Tensor] = None,
    projection_matrix: Optional[torch.Tensor] = None,
    device="cuda",
    ):
        """Given the image, depth, intrinsic and pose, creates a Camera object."""
        fx, fy, cx, cy = intrinsic
        height, width = image.shape[2:]
        gt_pose = torch.linalg.inv(gt_pose)
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)

        image = image.squeeze().to(device)
        depth = depth.to(device)
        gt_pose = gt_pose.to(device)
        projection_matrix = projection_matrix.to(device)
        dynamic_mask = dynamic_mask.to(device)

                
        return Camera(
            idx,
            image,
            depth,
            gt_pose,
            projection_matrix,
            fx,
            fy,
            cx,
            cy,
            fovx,
            fovy,
            height,
            width,
            device=device,
            dyn_mask=dynamic_mask,
        )


def lunch_gui(cfg, queue, gaussians, background, gui_done):
    params_gui = gui_utils.ParamsGUI(
        pipe=cfg.mapping.pipeline_params,
        background=background,
        gaussians=gaussians,
        q_main2vis=queue,
    )
    end = False
    while not end:
        end = slam_gui.run(params_gui)
    gui_done += 1
    return 

    

@hydra.main(version_base=None, config_path="./configs/", config_name="slam")
def main(cfg):
    dataset = get_dataset(cfg, device=cfg.device)
    torch.multiprocessing.set_start_method("spawn")
    kf_mng_params = cfg.mapping.keyframes.mapping
    sh_degree = 3 if cfg.mapping.use_spherical_harmonics else 0

    #gaussians = GaussianModel(sh_degree, config=cfg.mapping.input)
    gaussians = DynamicGaussianModel(sh_degree, lifespan=100, config=cfg.mapping.input)

    gaussians.init_lr(cfg.mapping.opt_params.init_lr)
    gaussians.training_setup(cfg.mapping.opt_params)

    q_main2vis = mp.Queue()
    gui_done = torch.zeros((1)).int().share_memory_()

    bg_color = [1, 1, 1]  # White background
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    processes = mp.Process(target=lunch_gui, args=(cfg, q_main2vis, gaussians, background, gui_done), name="Mapping GUI",)
    processes.start()
    time.sleep(3)

    z_near = 0.001
    z_far = 100000.0

    timestamp, image, depth, intrinsic, gt_pose, dyn_mask = dataset[0]

    width, height = image.shape[2], image.shape[1]
    fx, fy, cx, cy = intrinsic

    projection_mat = getProjectionMatrix2(z_near, z_far, cx, cy, fx, fy, width, height).transpose(0, 1)

    cam = camera_from_frame(timestamp, image, depth, intrinsic, gt_pose, dyn_mask, projection_mat, device="cuda")

    gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)
    print("Gaussians: ", len(gaussians))

    render_pkg = render_dynamic(cam, gaussians, cfg.mapping.pipeline_params, background, device=cfg.device)
        # #render_pkg = render_dynamic(cam, gaussians, cfg.mapping.pipeline_params, background, device=cfg.device, time=0)
    rgb = np.uint8(255 * render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        #cv2.imshow("Render", bgr)
    cv2.imwrite("/home/leon/test1.png", bgr)

    for i in range(1000):
        print("Iteration: {}/1000".format(i+1), end="\r")
        render_pkg = render_dynamic(cam, gaussians, cfg.mapping.pipeline_params, background, device=cfg.device)


        loss = torch.abs(render_pkg["render"] - cam.original_image).sum()
        loss.backward()

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad()

            if i % 20 == 0:
                gaussians.add_densification_stats(0, render_pkg["viewspace_points"], render_pkg["visibility_filter"])

                gaussians.densify_and_prune(  # General pruning based on opacity and size + densification
                        0,
                        kf_mng_params.densify_grad_threshold*2,
                        kf_mng_params.opacity_th,
                        kf_mng_params.gaussian_extent,
                        kf_mng_params.size_threshold,
                    )


        q_main2vis.put_nowait(
            gui_utils.GaussianPacket(
                gaussians=clone_obj(gaussians),
                current_frame=cam,
                keyframes=[cam],
                #gtcolor=cam.original_image,
                gtcolor=render_pkg["render"].detach(),
                gtdepth=cam.depth.cpu().numpy(),
                gaussian_type="dynamic",
            )
        )

    #render_pkg = render(cam, gaussians, cfg.mapping.pipeline_params, background, device=cfg.device)
    render_pkg = render_dynamic(cam, gaussians, cfg.mapping.pipeline_params, background, device=cfg.device, time=0)
    rgb = np.uint8(255 * render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        #cv2.imshow("Render", bgr)
    cv2.imwrite("/home/leon/test2.png", bgr)



    while gui_done < 1:
        pass

    print("Done")


if __name__ == "__main__":
    main()