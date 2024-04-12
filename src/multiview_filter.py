from termcolor import colored

import torch
import torch.nn as nn
import torch.nn.functional as F

import droid_backends
from lietorch import SE3

"""
Filter multiple views to remove occluded points and outliers.

This checks for consistency, i.e. a 3D point with a known 2D correspondence across multiple views should be reprojected 
to roughly the same 2D coordinate when reprojecting. The depth_filter function checks for this by counting how many 
points are falling inside the same threshold bin. If there are not enough points consistent across multiple views, they will be filtered out. 

NOTE you can use this filter as a good density proxy for cleaning noisy point clouds
"""


class MultiviewFilter(nn.Module):
    def __init__(self, cfg, args, slam):
        super(MultiviewFilter, self).__init__()

        self.args = args
        self.cfg = cfg
        self.device = args.device
        self.warmup = cfg["tracking"]["warmup"]
        # dpeth error < 0.01m
        self.filter_thresh = cfg["tracking"]["multiview_filter"]["thresh"]
        # points viewed by at least 3 cameras
        self.filter_visible_num = cfg["tracking"]["multiview_filter"]["visible_num"]
        self.kernel_size = cfg["tracking"]["multiview_filter"]["kernel_size"]  # 3
        self.bound_enlarge_scale = cfg["tracking"]["multiview_filter"]["bound_enlarge_scale"]
        self.net = slam.net
        self.video = slam.video
        self.mode = slam.mode

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    # NOTE chen: This distance can already be computed with a single function call in Lietorch
    def pose_dist(self, Tquad0: torch.Tensor, Tquad1: torch.Tensor) -> torch.Tensor:
        # Tquad with shape [batch_size, 7]
        def quat_to_euler(Tquad: torch.Tensor) -> torch.Tensor:
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)

            args:
            ---
            Tquad [Tensor]: quaternion with shape [batch_size, 7]
            """
            tx, ty, tz, x, y, z, w = torch.unbind(Tquad, dim=-1)
            t0 = 2.0 * (w * x + y * z)
            t1 = 1.0 - 2.0 * (x * x + y * y)
            roll_x = torch.atan2(t0, t1)

            t2 = 2.0 * (w * y - z * x)
            t2 = torch.clamp(t2, min=-1.0, max=1.0)
            pitch_y = torch.asin(t2)

            t3 = 2.0 * (w * z + x * y)
            t4 = 1.0 - 2.0 * (y * y + z * z)
            yaw_z = torch.atan2(t3, t4)

            Teuler = torch.stack([tx, ty, tz, roll_x, pitch_y, yaw_z], dim=-1)

            return Teuler

        # Refer to BundleFusion Sec5.3
        Teuler0 = quat_to_euler(Tquad0)
        Teuler1 = quat_to_euler(Tquad1)
        dist = (Teuler0 - Teuler1).abs()
        return 1.0 * dist[:, :3].sum(dim=-1) + 2.0 * dist[:, 3:].sum(dim=-1)

    @torch.no_grad()
    def in_bound(self, pts: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
        """
        Check which points are inside the bounding box.

        Args:
        ---
        pts [Tensor]:  3d points of shape [n_points, 3]
        bound [Tensor]: bound of shape [3, 2]
        """
        bound = bound.to(pts.device)
        mask_x = (pts[:, 0] < bound[0, 1]) & (pts[:, 0] > bound[0, 0])
        mask_y = (pts[:, 1] < bound[1, 1]) & (pts[:, 1] > bound[1, 0])
        mask_z = (pts[:, 2] < bound[2, 1]) & (pts[:, 2] > bound[2, 0])
        return (mask_x & mask_y & mask_z).bool()

    @torch.no_grad()
    def get_bound_from_pointcloud(self, pts: torch.Tensor, enlarge_scale: float = 1.0) -> torch.Tensor:
        bound = torch.stack(
            [torch.min(pts, dim=0, keepdim=False).values, torch.max(pts, dim=0, keepdim=False).values], dim=-1
        )
        enlarge_bound_length = (bound[:, 1] - bound[:, 0]) * (enlarge_scale - 1.0)
        # extend max 1.0m on boundary
        # enlarge_bound_length = torch.min(enlarge_bound_length, torch.ones_like(enlarge_bound_length) * 1.0)
        bound_edge = torch.stack([-enlarge_bound_length / 2.0, enlarge_bound_length / 2.0], dim=-1)
        bound = bound + bound_edge

        return bound

    def info(self, bound: torch.Tensor, masks: torch.Tensor, filtered_t: int, cur_t: int) -> None:
        prefix = "Bound: ["
        bd = bound.tolist()
        prefix += f"[{bd[0][0]:.1f}, {bd[0][1]:.1f}], "
        prefix += f"[{bd[1][0]:.1f}, {bd[1][1]:.1f}], "
        prefix += f"[{bd[2][0]:.1f}, {bd[2][1]:.1f}]]!"

        msg = "\n\n Multiview filtering: previous at {}, now at {}, {} valid points found! {}\n".format(
            filtered_t, cur_t, masks.sum(), prefix
        )
        print(colored("[Multiview Filter]: " + msg, "cyan"))

    def forward(self):
        """Filter out occluded, outliers and low density points using multiview consistency"""

        cur_t = self.video.counter.value
        filtered_t = int(self.video.filtered_id.item())
        if filtered_t < cur_t and cur_t > self.warmup:

            # NOTE chen: we keep many computations inside the lock, because this used to raise a malloc error on my setup if not done
            with self.video.get_lock():
                dirty_index = torch.arange(0, cur_t).long().to(self.device)
                poses = torch.index_select(self.video.poses.detach(), dim=0, index=dirty_index)
                disps = torch.index_select(self.video.disps_up.detach(), dim=0, index=dirty_index)
                common_intrinsic_id = 0  # we assume the intrinsics are consistent within one scene
                intrinsic = self.video.intrinsics[common_intrinsic_id].detach() * self.video.scale_factor
                w2w = SE3(self.video.pose_compensate[0].clone().unsqueeze(dim=0)).to(self.device)

                points = droid_backends.iproj((w2w * SE3(poses).inv()).data, disps, intrinsic)  # [b, h, w 3]
                thresh = self.filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
                count = droid_backends.depth_filter(poses, disps, intrinsic, dirty_index, thresh)  # [b, h, w]

            bs, ht, wd = count.shape
            # We only keep points that are consistent across multiple views within a threshold
            is_consistent = count >= self.filter_visible_num
            # filter out far points
            is_consistent = is_consistent & (disps > 0.01 * disps.mean(dim=[1, 2], keepdim=True))
            if is_consistent.sum() < 100:  # Do not filter away small scenes
                return
            bound = self.get_bound_from_pointcloud(points.reshape(-1, 3)[is_consistent.reshape(-1)])  # [3, 2]

            if isinstance(self.kernel_size, str) and self.kernel_size == "inf":
                is_consistent_padded = torch.ones_like(is_consistent).bool()
            elif int(self.kernel_size) < 2:
                is_consistent_padded = is_consistent
            else:
                kernel = int(self.kernel_size)
                kernel = (kernel // 2) * 2 + 1  # odd number

                is_consistent_padded = (
                    F.conv2d(
                        is_consistent.unsqueeze(dim=1).float(),
                        weight=torch.ones(1, 1, kernel, kernel, dtype=torch.float, device=is_consistent.device),
                        stride=1,
                        padding=kernel // 2,
                        bias=None,
                    )
                    .bool()
                    .squeeze(dim=1)
                )  # [b, h, w]

            if is_consistent_padded.sum() < 100:  # Do not filter away small scenes
                return

            is_in_bounds = self.in_bound(points.reshape(-1, 3), bound)  # N'
            valid = torch.logical_and(is_in_bounds, is_consistent_padded.reshape(-1))
            points_filtered = points.reshape(-1, 3)[valid.reshape(-1)]
            new_bound = self.get_bound_from_pointcloud(points_filtered)

            # cur_t is the current last visited frame in the video
            # TODO chen: what do we do with these?
            priority = self.pose_dist(self.video.poses_filtered[:cur_t].detach(), poses)

            with self.video.mapping.get_lock():
                self.video.update_priority[:cur_t] += priority.detach()
                self.video.mask_filtered[:cur_t] = valid.reshape(bs, ht, wd).detach()
                self.video.disps_filtered[:cur_t] = disps.detach()
                self.video.poses_filtered[:cur_t] = poses.detach()
                # Update the filter id
                self.video.filtered_id[0] = cur_t
                self.video.bound[0] = new_bound

            self.info(new_bound, is_consistent, filtered_t, cur_t)
            del points, is_consistent, poses, disps
            torch.cuda.empty_cache()
