import torch
import torch.nn as nn
import droid_backends
from lietorch import SE3
from colorama import Fore, Style
import torch.nn.functional as F


class MultiviewFilter(nn.Module):
    def __init__(self, cfg, slam):
        super(MultiviewFilter, self).__init__()

        self.cfg = cfg
        self.device = cfg.slam.device
        self.warmup = cfg["tracking"]["warmup"]
        self.filter_thresh = cfg["tracking"]["multiview_filter"]["thresh"]  # depth error < 0.01m
        # points viewed by at least 3 cameras
        self.filter_visible_num = cfg["tracking"]["multiview_filter"]["visible_num"]
        self.kernel_size = cfg["tracking"]["multiview_filter"]["kernel_size"]  # 3
        self.bound_enlarge_scale = cfg["tracking"]["multiview_filter"]["bound_enlarge_scale"]
        self.net = slam.net
        self.video = slam.video
        self.mode = slam.mode

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def pose_dist(self, Tquad0, Tquad1):
        """Compute the distance between two quaternions

        returns:
        ---
        dist [torch.Tensor]: Distance tensor of shape [batch_size, ]
        """

        def quat_to_euler(Tquad):
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
    def in_bound(self, pts, bound):
        """
        Args:
        ---
        pts [Tensor]:  3d points of shape [n_points, 3]
        bound [Tensor]: bound of shape [3, 2]
        """
        bound = bound.to(pts.device)  # mask for points out of bound
        mask_x = (pts[:, 0] < bound[0, 1]) & (pts[:, 0] > bound[0, 0])
        mask_y = (pts[:, 1] < bound[1, 1]) & (pts[:, 1] > bound[1, 0])
        mask_z = (pts[:, 2] < bound[2, 1]) & (pts[:, 2] > bound[2, 0])
        return (mask_x & mask_y & mask_z).bool()

    @torch.no_grad()
    def get_bound_from_pointcloud(self, pts, enlarge_scale=1.0):
        bound = torch.stack(
            [torch.min(pts, dim=0, keepdim=False).values, torch.max(pts, dim=0, keepdim=False).values], dim=-1
        )  # [3, 2]
        enlarge_bound_length = (bound[:, 1] - bound[:, 0]) * (enlarge_scale - 1.0)
        # extend max 1.0m on boundary
        # enlarge_bound_length = torch.min(enlarge_bound_length, torch.ones_like(enlarge_bound_length) * 1.0)
        bound_edge = torch.stack([-enlarge_bound_length / 2.0, enlarge_bound_length / 2.0], dim=-1)
        return bound + bound_edge

    def forward(self):
        cur_t = self.video.counter.value
        filtered_t = int(self.video.filtered_id.item())

        if filtered_t < cur_t and cur_t > self.warmup:
            # NOTE chen: put more computations into this lock context, else some memory (malloc()) errors are triggered on my setup
            with self.video.get_lock():
                dirty_index = torch.arange(0, cur_t).long().to(self.device)
                poses = torch.index_select(self.video.poses.detach(), dim=0, index=dirty_index)
                disps = torch.index_select(self.video.disps_up.detach(), dim=0, index=dirty_index)
                common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
                intrinsic = self.video.intrinsics[common_intrinsic_id].detach() * self.video.scale_factor
                w2w = SE3(self.video.pose_compensate[0].clone().unsqueeze(dim=0)).to(self.device)

                points = droid_backends.iproj((w2w * SE3(poses).inv()).data, disps, intrinsic)  # [b, h, w 3]
                thresh = self.filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
                count = droid_backends.depth_filter(poses, disps, intrinsic, dirty_index, thresh)  # [b, h, w]

            masks = count >= self.filter_visible_num
            # filter out far points, [b, h, w]
            masks = masks & (disps > 0.01 * disps.mean(dim=[1, 2], keepdim=True))
            if masks.sum() < 100:
                return
            sel_points = points.reshape(-1, 3)[masks.reshape(-1)]
            bound = self.get_bound_from_pointcloud(sel_points)  # [3, 2]

            if isinstance(self.kernel_size, str) and self.kernel_size == "inf":
                extended_masks = torch.ones_like(masks).bool()
            elif int(self.kernel_size) < 2:
                extended_masks = masks
            else:
                kernel = int(self.kernel_size)
                kernel = (kernel // 2) * 2 + 1  # odd number

                extended_masks = (
                    F.conv2d(
                        masks.unsqueeze(dim=1).float(),
                        weight=torch.ones(1, 1, kernel, kernel, dtype=torch.float, device=masks.device),
                        stride=1,
                        padding=kernel // 2,
                        bias=None,
                    )
                    .bool()
                    .squeeze(dim=1)
                )  # [b, h, w]

            if extended_masks.sum() < 100:
                return
            sel_points = points.reshape(-1, 3)[extended_masks.reshape(-1)]
            in_bound_mask = self.in_bound(sel_points, bound)  # N'
            # FIXME This triggeres a bug when doing mapping, filtering, visualizing, optimizing, etc. together
            # RuntimeError: out_ptr == out_accessor[thread_count_nonzero[tid + 1]].data()
            # INTERNAL ASSERT FAILED at "/opt/conda/conda-bld/pytorch_1702400430266/work/aten/src/ATen/native/TensorAdvancedIndexing.cpp":2336, please report a bug to PyTorch
            extended_masks[extended_masks.clone()] = in_bound_mask

            sel_points = points.reshape(-1, 3)[extended_masks.reshape(-1)]
            bound = self.get_bound_from_pointcloud(sel_points)  # [3, 2]
            priority = self.pose_dist(self.video.poses_filtered[:cur_t].detach(), poses)

            with self.video.mapping.get_lock():
                self.video.update_priority[:cur_t] += priority.detach()
                self.video.mask_filtered[:cur_t] = extended_masks.detach()
                self.video.disps_filtered[:cur_t] = disps.detach()
                self.video.poses_filtered[:cur_t] = poses.detach()
                # Update the filter id
                self.video.filtered_id[0] = cur_t
                self.video.bound[0] = bound

            prefix = "Bound: ["
            bd = bound.tolist()
            prefix += f"[{bd[0][0]:.1f}, {bd[0][1]:.1f}], "
            prefix += f"[{bd[1][0]:.1f}, {bd[1][1]:.1f}], "
            prefix += f"[{bd[2][0]:.1f}, {bd[2][1]:.1f}]]!"
            print(Fore.CYAN)
            msg = f"\n\n Multiview filtering: previous at {filtered_t}, now at {cur_t}, {masks.sum()} valid points found! {prefix}\n"
            print(msg)
            print(Style.RESET_ALL)

            del points, masks, poses, disps
            torch.cuda.empty_cache()
