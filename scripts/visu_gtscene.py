#! /usr/bin/env python3


"""
Go over a data stream with existing groundtruth poses, depth and object masks 
and visualize the scene in 3D. 

For dyn. objects, the mask is used to remove/add the object from the scene for each frame.
"""


import ipdb
from copy import deepcopy
from tqdm import tqdm
from typing import Optional, Union
from termcolor import colored
from time import sleep
import ipdb

import hydra

import torch
from torch.multiprocessing import Value, Queue, Process

from src.datasets import get_dataset
from src.geom import matrix_to_lie
from src.visualization import CAM_LINES, CAM_POINTS, create_camera_actor, create_point_actor

from lietorch import SE3
import droid_backends

import open3d as o3d


class SimpleVideo:
    """
    Data structure of multiple buffers to keep track of indices, poses, disparities, images, etc.

    This is useful if we want to manipulate the scene on-the-fly, i.e. we might filter interactively instead of just
    iterating over the dataset.
    """

    def __init__(
        self,
        ht: int,
        wd: int,
        device: str = "cuda:0",
        buffer: int = 512,
        has_static_mask: bool = False,
        downscale: Optional[int] = 4,
    ):

        self.counter = Value("i", 0)

        ### Intrinsics / Calibration ###
        # Pinhole model
        self.n_intr = 4
        self.model_id = 0

        self.ht = ht
        self.wd = wd
        self.device = device

        self.s = downscale

        self.buffer = buffer
        self.upsampled = True

        ### State attributes ###
        self.timestamp = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()

        self.has_static_mask = has_static_mask
        if self.s is not None:
            self.images = torch.zeros(buffer, 3, ht // self.s, wd // self.s, device=device, dtype=torch.float)
            self.disps = torch.zeros(
                buffer, ht // self.s, wd // self.s, device=device, dtype=torch.float
            ).share_memory_()
            self.static_masks = torch.ones(
                buffer, ht // self.s, wd // self.s, device=device, dtype=torch.bool
            ).share_memory_()
        else:
            self.images = torch.zeros(buffer, 3, ht, wd, device=device, dtype=torch.float)
            self.disps = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
            self.static_masks = torch.ones(buffer, ht, wd, device=device, dtype=torch.bool).share_memory_()

        self.intrinsics = torch.zeros(buffer, 4, device=device, dtype=torch.float).share_memory_()

        self.poses = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # w2c quaterion
        self.poses_gt = torch.zeros(buffer, 4, 4, device=device, dtype=torch.float).share_memory_()  # c2w matrix
        # Initialize poses to identity transformation
        self.poses[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.poses_gt[:] = torch.eye(4, dtype=torch.float, device=device)

    def get_lock(self):
        return self.counter.get_lock()

    def __len__(self):
        return self.buffer

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.timestamp[index] = item[0]
        image = item[1]
        if self.s is not None:
            image = image[..., int(self.s // 2 - 1) :: self.s, int(self.s // 2 - 1) :: self.s]
        self.images[index] = image

        if item[2] is not None:
            poses_gt = item[2].to(self.device)
            self.poses_gt[index] = poses_gt
            self.poses[index] = matrix_to_lie(poses_gt).inv().vec()  # [7, 1] Lie element

        if item[3] is not None:
            depth = item[3]
            if self.s is not None:
                depth = depth[..., int(self.s // 2 - 1) :: self.s, int(self.s // 2 - 1) :: self.s]
            self.disps[index] = torch.where(depth > 0, 1.0 / depth, depth)

        if item[4] is not None:
            intrinsics = item[4]
            # NOTE we downsample the image, so adjust intrinsics
            if self.s is not None:
                intrinsics = intrinsics / self.s
            self.intrinsics[index] = intrinsics

        if len(item) > 5 and item[5] is not None:
            mask = item[5]
            if self.s is not None:
                mask = mask[..., int(self.s // 2 - 1) :: self.s, int(self.s // 2 - 1) :: self.s]
            self.static_masks[index] = mask

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """index the depth video"""

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index > 0:
                index = self.counter.value + index
            item = (
                self.images[index],
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
            )
            if self.has_static_mask:
                item += (self.static_masks[index],)

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


def droid_visualization(video, device="cuda:0", q_vis2main: Optional[Queue] = None, with_dynamic: bool = False):
    """DROID visualization frontend"""

    droid_visualization.sleep_delay = 0.1
    droid_visualization.wait = False
    # Use the Queue to feedback signals from GUI to main logic
    droid_visualization.queue = q_vis2main

    droid_visualization.with_dynamic = with_dynamic
    if with_dynamic:
        video_has_dyn = video.static_masks.sum() != video.static_masks.numel()
        assert video_has_dyn, "Dataset does not contain dynamic masks, no dynamics to visualize!"

    torch.cuda.set_device(device)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    # NOTE chen: normally we might have multiple moving objects in the scene
    # however we currently just store the whole foreground as one object and refresh this over time
    # NOTE chen: if you want to do this correctly you also need an additional state attribute for each object,
    # this would work best when keeping a buffer over time for each object and perform tracking
    # the reason for this is that we dont want to delete/refresh objects that stop moving, i.e. we would need to
    # detect these state-changes in the visualization as well
    droid_visualization.moving = {}

    droid_visualization.scale = 10.0  # 1.0
    droid_visualization.camera_scale = 0.025
    droid_visualization.ix = 0

    # Thresholds for visualization filtering
    droid_visualization.mv_filter_thresh = 0.001
    droid_visualization.mv_filter_count = 1

    droid_visualization.do_reset = True

    def increase_mv_filter(vis):
        droid_visualization.mv_filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def decrease_mv_filter(vis):
        droid_visualization.mv_filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def increase_mv_count(vis):
        droid_visualization.mv_filter_count += 1
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def decrease_mv_count(vis):
        # Count should be > 1!
        if droid_visualization.mv_filter_count > 1:
            droid_visualization.mv_filter_count -= 1
            with droid_visualization.video.get_lock():
                droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def deactivate_update(vis):
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:] = False

    def increase_camera(vis):
        droid_visualization.camera_scale *= 1 / 0.8
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def decrease_camera(vis):
        droid_visualization.camera_scale *= 0.8
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def start_stop_view_resetting(vis):
        droid_visualization.do_reset = not droid_visualization.do_reset

    def stop_processing(vis):
        droid_visualization.wait = True

    def continue_processing(vis):
        droid_visualization.wait = False

    @torch.no_grad()
    def animation_callback(vis):
        cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

        # Put this into the Queue, so the main loop can continue
        if not droid_visualization.wait:
            droid_visualization.queue.put("continue")

        # Check for updates
        with video.get_lock():
            t = video.counter.value
            (dirty_index,) = torch.where(video.dirty.clone())
            dirty_index = dirty_index

        # If no update, dont process
        if len(dirty_index) == 0:
            return

        ### Main Processing
        video.dirty[dirty_index] = False  # Reset dirty
        poses = torch.index_select(video.poses, 0, dirty_index)
        disps = torch.index_select(video.disps, 0, dirty_index)
        valid = disps > 0
        disps = disps.clamp_(min=0.001, max=1000.0)  # Avoid division by zero

        # convert poses to 4x4 matrix
        Ps = SE3(poses).inv().matrix().cpu().numpy()
        images = torch.index_select(video.images, 0, dirty_index)
        assert disps.shape[-2:] == images.shape[-2:], "Disps and images need to have the same shape"
        images = images.permute(0, 2, 3, 1).cpu()

        intrinsics = video.intrinsics[0]
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics).cpu()

        thresh = droid_visualization.mv_filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
        # Take global intrinsics, since they should be the same across the video!
        count = droid_backends.depth_filter(video.poses, video.disps, intrinsics, dirty_index, thresh)
        count, disps = count.cpu(), disps.cpu()
        # Only keep points that are consistent across multiple views and not too close by
        masks = (count >= droid_visualization.mv_filter_count) & (disps > 0.5 * disps.mean(dim=[1, 2], keepdim=True))
        masks = masks & valid.cpu()  # Remove invalid points (e.g. with 0 depth)

        if droid_visualization.with_dynamic:
            stat_masks = torch.index_select(video.static_masks, 0, dirty_index)
            masks = masks & stat_masks.cpu()

        # Go over dirty frames
        for i in range(len(dirty_index)):
            pose = Ps[i]
            ix = dirty_index[i].item()

            if droid_visualization.with_dynamic and len(droid_visualization.moving) > 0:
                # NOTE chen: right now we just keep one foreground object and refresh every time
                # Go over all past objects and refresh
                # TODO normally we would have to check for the state of frame-1 here
                for k, v in droid_visualization.moving.items():
                    vis.remove_geometry(v, reset_bounding_box=droid_visualization.do_reset)

            if ix in droid_visualization.cameras:
                vis.remove_geometry(
                    droid_visualization.cameras[ix],
                    reset_bounding_box=droid_visualization.do_reset,
                )
                del droid_visualization.cameras[ix]

            if ix in droid_visualization.points:
                vis.remove_geometry(
                    droid_visualization.points[ix],
                    reset_bounding_box=droid_visualization.do_reset,
                )
                del droid_visualization.points[ix]

            ### add camera actor ###
            cam_actor = create_camera_actor(True, droid_visualization.camera_scale)
            cam_actor.transform(pose)
            vis.add_geometry(cam_actor, reset_bounding_box=droid_visualization.do_reset)
            droid_visualization.cameras[ix] = cam_actor

            if droid_visualization.with_dynamic:
                dyn_masks = ~stat_masks[i].cpu()
                dyn_mask = dyn_masks.reshape(-1)
                dyn_pts = points[i].reshape(-1, 3)[dyn_mask].numpy()
                dyn_clr = images[i].reshape(-1, 3)[dyn_mask].numpy()

                ## Add point actor ##
                dyn_point_actor = create_point_actor(dyn_pts, dyn_clr)
                vis.add_geometry(dyn_point_actor, reset_bounding_box=droid_visualization.do_reset)
                droid_visualization.moving[ix] = dyn_point_actor

            mask = masks[i].reshape(-1)
            pts = points[i].reshape(-1, 3)[mask].numpy()
            clr = images[i].reshape(-1, 3)[mask].numpy()

            ## Add point actor ###
            point_actor = create_point_actor(pts, clr)
            vis.add_geometry(point_actor, reset_bounding_box=droid_visualization.do_reset)
            droid_visualization.points[ix] = point_actor
        ###

        droid_visualization.ix += 1
        cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, True)
        vis.poll_events()
        vis.update_renderer()

        # Put this into the Queue, so the main loop can wait for the signal
        if droid_visualization.wait:
            droid_visualization.queue.put("wait")

        # Let other processes catch up and cool down
        sleep(droid_visualization.sleep_delay)

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)

    ## Create keys
    vis.register_key_callback(ord("S"), increase_mv_filter)
    vis.register_key_callback(ord("A"), decrease_mv_filter)

    vis.register_key_callback(ord("V"), increase_mv_count)
    vis.register_key_callback(ord("B"), decrease_mv_count)

    vis.register_key_callback(ord("M"), increase_camera)
    vis.register_key_callback(ord("N"), decrease_camera)

    vis.register_key_callback(ord("X"), continue_processing)
    vis.register_key_callback(ord("C"), stop_processing)

    vis.register_key_callback(ord("Q"), deactivate_update)
    vis.register_key_callback(ord("R"), start_stop_view_resetting)
    vis.create_window(height=540, width=960)

    vis.get_render_option().load_from_json("src/renderoption.json")

    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()

    vis.destroy_window()
    return True


def sys_print(msg: str) -> None:
    print(colored(msg, "white", "on_grey", attrs=["bold"]))


def get_latest_element_of_queue(queue: Queue):
    if queue.empty():
        return None

    while not queue.empty():
        item = queue.get()
        dummy = deepcopy(item)
        del item
    return dummy


def test_scene(video: SimpleVideo, index: Union[int, torch.Tensor]) -> None:
    """Make a simple visualization of the scene at a given index. We simply backproject the scene
    into 3D given the disparity, camera poses and intrinsics and overlay with the image color.

    We can use this to debug the scene and see if the scene is correctly loaded before animating the whole video.
    """
    poses = torch.index_select(video.poses, 0, index)
    disps = torch.index_select(video.disps, 0, index)
    images = torch.index_select(video.images, 0, index)
    images = images.permute(0, 2, 3, 1)  # Put into numpy/o3d convention
    intrinsics = video.intrinsics[0]  # We have constant global intrinsics over video

    Ps = SE3(poses).inv().matrix().cpu().numpy()  # Convert to lietorch SE3
    points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics)

    masks = disps > 0  # Filter out invalid disparities (we mark nan's with 0)

    geometries = []
    for i in range(len(index)):
        pts = points[i][masks[i], :].reshape(-1, 3).cpu().numpy()
        clr = images[i][masks[i], :].reshape(-1, 3).cpu().numpy()
        point_actor = create_point_actor(pts, clr)

        cam_actor = create_camera_actor(True)
        cam_actor.transform(Ps[i])

        geometries.append(point_actor)
        geometries.append(cam_actor)

    # Plot the point cloud
    o3d.visualization.draw_geometries(geometries)


def show_masks_video(datastructure: SimpleVideo) -> None:
    """Use an animation to show the masks over time."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(1, 1)
    ax.axis("off")

    def update(i):
        mask = datastructure.static_masks[i].cpu().numpy()
        ax.imshow(mask)
        ax.set_title(f"Frame {i}")

    ani = FuncAnimation(fig, update, frames=len(datastructure), repeat=False)
    plt.show()


@hydra.main(version_base=None, config_path="../configs/", config_name="visu")
def main(cfg):

    output_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    torch.multiprocessing.set_start_method("spawn")

    dataset = get_dataset(cfg, device=cfg.device)
    sys_print(f"Running on {len(dataset)} frames")
    with_dyn = cfg.get("with_dyn", True)
    t_start = cfg.get("t_start", 0)
    t_stop = cfg.get("t_stop", len(dataset))

    # NOTE here we store the whole scene in memory, which is excessive
    # watch out that this does not OOM (we are fine on a 4090 and ~2000 frames, which is a lot already)
    sample_image = dataset[0][1]
    len_data_snippet = t_stop - t_start
    datastructure = SimpleVideo(
        ht=sample_image.shape[-2],
        wd=sample_image.shape[-1],
        device=cfg.device,
        buffer=min(len_data_snippet, len(dataset)),
        downscale=cfg.get("downscale", None),
        has_static_mask=(with_dyn and dataset.has_dyn_masks),
    )

    i = 0
    for frame in tqdm(dataset):
        if i < t_start:
            i += 1
            continue

        if i >= t_stop:
            break

        if dataset.has_dyn_masks and with_dyn:
            timestamp, image, depth, intrinsic, gt_pose, stat_mask = frame
            datastructure[i] = (timestamp, image, gt_pose, depth, intrinsic, stat_mask)
        else:
            timestamp, image, depth, intrinsic, gt_pose = frame
            datastructure[i] = (timestamp, image, gt_pose, depth, intrinsic)
        datastructure.counter.value += 1
        i += 1

    n_frames = datastructure.counter.value - 1
    datastructure.counter.value = 0  # Reset counter

    # test_scene(datastructure, torch.arange(0, 3, device=datastructure.device)) # Is the reconstruction of individual frames correct?
    # show_masks_video(datastructure) # How do the masks actually look like?

    # Initialize the visualization process
    q_vis2main = Queue()  # Use a Queue to stop iterating over dataset if wanted triggered from GUI

    show_dynamics = dataset.has_dyn_masks and with_dyn
    sys_print(f"Visualizing with dynamics: {show_dynamics}")
    visualization = Process(
        target=droid_visualization,
        args=(datastructure, cfg.device, q_vis2main, show_dynamics),
    )
    visualization.start()

    # Go over the video and set frames as dirty
    for i in tqdm(range(n_frames)):
        if i > t_stop:
            break

        # Get latest signal from Queue to check if we can keep going
        signal = get_latest_element_of_queue(q_vis2main)
        if signal is not None and signal == "wait":
            sys_print("Waiting for signal to continue...")
            while True:
                signal = get_latest_element_of_queue(q_vis2main)
                if signal is not None and signal == "continue":
                    sys_print("Continuing...")
                    break

        with datastructure.get_lock():
            datastructure.dirty[i] = True
            datastructure.counter.value += 1

        if i == 0:
            sleep(cfg.get("sleep_init", 5.0))  # Let window pop up first
        sleep(cfg.get("delay", 0.5))

    visualization.join()
    sys_print("Done!")


if __name__ == "__main__":
    main()
