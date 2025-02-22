import glob
import os
import ipdb
from omegaconf import DictConfig
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch
import liblzfse
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from kornia.geometry.linalg import compose_transformations

from .geom import matrix_to_lie


def readEXR_onlydepth(filename: str):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y


def read_sintel_depth(filename: str) -> np.ndarray:
    """Read depth data from file, return as numpy array."""

    tag_float = 202021.25
    tag_char = "PIEH"

    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == tag_float
    ), " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(tag_float, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert (
        width > 0 and height > 0 and size > 1 and size < 100000000
    ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(width, height)
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_dataset(cfg: DictConfig, device="cuda:0"):
    return dataset_dict[cfg.data.dataset](cfg, device=device)


class BaseDataset(Dataset):
    """Barebone dataset structure. This can load images and depth maps from specific folders.
    We also load potential masks, e.g. for dynamic objects if provided.

    Camera intrinsics are assumed to be passed in the config args.
    We assume a pinhole camera model, but also allow distortion coefficients to be passed.
    """

    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(BaseDataset, self).__init__()
        self.name = cfg.data.dataset
        self.stereo = cfg.mode == "stereo"
        self.device = device
        self.png_depth_scale = cfg.data.get("png_depth_scale", None)
        self.stride = cfg.get("stride", 1)
        self.mono_model = cfg.get("mono_depth", "metric3d-vit_giant2")

        self.input_folder = cfg.data.input_folder
        self.color_paths = sorted(glob.glob(self.input_folder))

        # Manage the stream, i.e. we can also run the system only in [t0, t1]
        self.t_start = cfg.get("t_start", 0)
        self.t_stop = cfg.get("t_stop", None)
        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]

        self.color_paths = self.color_paths[:: self.stride]

        self.has_dyn_masks = False
        self.background_value = 0
        self.dilate_masks = True  # NOTE chen: some datasets (e.g. Sintel) have masks too small
        self.dilation_kernel_size = 1
        self.return_stat_masks = cfg.get("with_dyn", False)
        self.n_img = len(self.color_paths)

        self.depth_paths = None
        self.mask_paths = None
        self.poses = None
        self.relative_poses = False  # True to test the gt stream mapping

        self.image_timestamps = None

        self.H, self.W = int(cfg.data.cam.H), int(cfg.data.cam.W)
        self.H_out, self.W_out = int(cfg.data.cam.H_out), int(cfg.data.cam.W_out)
        self.fx, self.fy = float(cfg.data.cam.fx), float(cfg.data.cam.fy)
        self.cx, self.cy = float(cfg.data.cam.cx), float(cfg.data.cam.cy)
        self.H_edge, self.W_edge = int(cfg.data.cam.H_edge), int(cfg.data.cam.W_edge)

        self.distortion = np.array(cfg.data.cam.distortion) if "distortion" in cfg.data.cam else None

    def __len__(self) -> int:
        return self.n_img

    def depthloader(self, index: int) -> np.ndarray:
        if self.depth_paths is None:
            return None
        depth_path = self.depth_paths[index]
        if ".png" in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth_data /= self.png_depth_scale
        elif ".exr" in depth_path:
            depth_data = readEXR_onlydepth(depth_path).astype(np.float32)
            depth_data /= self.png_depth_scale
        elif ".npy" in depth_path:
            depth_data = np.load(depth_path).astype(np.float32)

        elif ".depth" in depth_path:  # NOTE leon: totalrecon depth files
            with open(depth_path, "rb") as depth_fh:
                raw_bytes = depth_fh.read()
                decompressed_bytes = liblzfse.decompress(raw_bytes)
                depth_data = np.frombuffer(decompressed_bytes, dtype=np.float32)
                depth_data = np.copy(depth_data.reshape(256, 192))  # NOTE leon: their depth shape
        elif ".dpt" in depth_path:  # NOTE chen: Sintel depth files
            depth_data = read_sintel_depth(depth_path)
        else:
            raise TypeError(depth_path)

        # Sanity check, because some datasets did not clean their depths
        depth_data[np.isnan(depth_data)] = 0.0

        return depth_data

    def _get_image(self, index: int) -> torch.Tensor:
        color_path = self.color_paths[index]
        color_data = cv2.imread(color_path)

        H_out_with_edge, W_out_with_edge = (
            self.H_out + self.H_edge * 2,
            self.W_out + self.W_edge * 2,
        )
        color_data = cv2.resize(color_data, (W_out_with_edge, H_out_with_edge))
        color_data = (
            torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0
        )  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, C, H, W]

        # crop image edge, there are invalid value on the edge of the color image
        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]

        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]

        return color_data

    def _get_depth(self, index: int) -> torch.Tensor:
        depth_data = self.depthloader(index)
        H_out_with_edge, W_out_with_edge = (self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2)
        outsize = (H_out_with_edge, W_out_with_edge)

        if depth_data is not None:
            depth_data = torch.from_numpy(depth_data).float()
            depth_data = F.interpolate(depth_data[None, None], outsize, mode="nearest")[0, 0]
            # Crop
            if self.H_edge > 0:
                edge = self.H_edge
                depth_data = depth_data[edge:-edge, :]
            if self.W_edge > 0:
                edge = self.W_edge
                depth_data = depth_data[:, edge:-edge]
        return depth_data

    def __getitem__(self, index: int):
        color_path = self.color_paths[index]
        color_data = cv2.imread(color_path)

        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx, self.cx, self.fy, self.cy
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        H_out_with_edge, W_out_with_edge = (self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2)
        outsize = (H_out_with_edge, W_out_with_edge)

        color_data = cv2.resize(color_data, (W_out_with_edge, H_out_with_edge))
        # bgr -> rgb, [0, 1]

        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        depth_data = self.depthloader(index)
        if depth_data is not None:
            depth_data = torch.from_numpy(depth_data).float()
            depth_data = F.interpolate(depth_data[None, None], outsize, mode="nearest")[0, 0]
            # Crop
            if self.H_edge > 0:
                edge = self.H_edge
                depth_data = depth_data[edge:-edge, :]
            if self.W_edge > 0:
                edge = self.W_edge
                depth_data = depth_data[:, edge:-edge]

        intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H

        # crop image edge, there are invalid value on the edge of the color image
        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
            intrinsic[3] -= edge

        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
            intrinsic[2] -= edge

        if self.poses is not None:
            pose = matrix_to_lie(torch.tensor(self.poses[index])).float()
        else:
            pose = None

        if self.has_dyn_masks and self.mask_paths is not None:
            # NOTE chen: in case we have multiple objects this mask could have range [0, 255]
            # -> TODO: in this case create array of masks with index 0 static scene, and others individual dyn. masks?
            # NOTE right now we only store a single static scene mask
            mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
            mask = mask == self.background_value  # static mask
            if self.dilate_masks:
                mask = np.uint8(~mask)  # Invert to dynamic mask
                kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                mask = ~mask.astype(bool)  # Return back to static mask
            mask = np.uint8(mask)
            mask = cv2.resize(mask, (W_out_with_edge, H_out_with_edge))
            mask = torch.from_numpy(mask).bool()
            if self.H_edge > 0:
                mask = mask[edge:-edge, :]
            if self.W_edge > 0:
                mask = mask[:, edge:-edge]

        if self.return_stat_masks:
            if not self.has_dyn_masks:
                raise Warning(
                    "Warning. Dataset does not have any dynamic masks, please provide some if you want to return them!"
                )
            return index, color_data, depth_data, intrinsic, pose, mask
        else:
            return index, color_data, depth_data, intrinsic, pose


class ImageFolder(BaseDataset):
    """Basic dataset for generic folder structures. Use this for custom datasets,
    e.g. videos you record with your phone. We assume the following folder structure:

    root/
        images/
        depths/ (optional)
        calib.txt (optional)
    """

    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(ImageFolder, self).__init__(cfg, device)
        # Get either jpg or png files
        input_images = os.path.join(self.input_folder, "images", "*.jpg")
        self.color_paths = sorted(glob.glob(input_images))
        # Look for alternative image extensions
        if len(self.color_paths) == 0:
            input_images = os.path.join(self.input_folder, "images", "*.png")
            self.color_paths = sorted(glob.glob(input_images))

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
        self.color_paths = self.color_paths[:: self.stride]

        depth_paths = sorted(glob.glob(os.path.join(self.input_folder, self.mono_model, "*.npy")))
        if len(depth_paths) != 0:
            self.depth_paths = depth_paths
            if self.t_stop is not None:
                self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[:: self.stride]
            assert len(self.depth_paths) == len(
                self.color_paths
            ), "Number of depth maps does not match number of images"

        self.n_img = len(self.color_paths)
        assert self.n_img > 0, f"No images found in {self.input_folder}"


class Replica(BaseDataset):
    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(Replica, self).__init__(cfg, device)
        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, "results/frame*.jpg")))
        # Set number of images for loading poses
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, "traj.txt"))

        # For Pseudo RGBD, we use monocular depth predictions in another folder
        if cfg.mode == "prgbd":
            self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, self.mono_model, "frame*.npy")))
            assert (
                len(self.depth_paths) == self.n_img
            ), f"Number of depth maps {len(self.depth_paths)} does not match number of images {self.n_img}"
        else:
            self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "results/depth*.png")))

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.poses = self.poses[self.t_start : self.t_stop]

        self.color_paths = self.color_paths[:: self.stride]
        self.depth_paths = self.depth_paths[:: self.stride]
        self.poses = self.poses[:: self.stride]

        if self.relative_poses:
            self.poses = torch.from_numpy(np.array(self.poses))
            trans_10 = torch.inverse(self.poses[0].unsqueeze(0).repeat(self.poses.shape[0], 1, 1))
            self.poses = compose_transformations(trans_10, self.poses).numpy()

        # Adjust number of images according to strides
        self.n_img = len(self.color_paths)

    def switch_to_rgbd_gt(self):
        """When evaluating, we want to use the ground truth depth maps."""
        self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "results/depth*.png")))
        self.depth_paths = self.depth_paths[:: self.stride]

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w)


class TartanAir(BaseDataset):
    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(TartanAir, self).__init__(cfg, device)
        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, "image_left/*.png")))
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, "pose_left.txt"))

        # For Pseudo RGBD, we use monocular depth predictions in another folder
        if cfg.mode == "prgbd":
            self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, self.mono_model + "_left", "*.npy")))
            assert (
                len(self.depth_paths) == self.n_img
            ), f"Number of depth maps {len(self.depth_paths)} does not match number of images {self.n_img}"
        else:
            self.depth_paths = None

        if self.depth_paths is not None:
            if self.t_stop is not None:
                self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[:: self.stride]

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
            self.poses = self.poses[self.t_start : self.t_stop]
        self.color_paths = self.color_paths[:: self.stride]
        self.poses = self.poses[:: self.stride]

        if self.relative_poses:
            self.poses = torch.from_numpy(np.array(self.poses))
            trans_10 = torch.inverse(self.poses[0].unsqueeze(0).repeat(self.poses.shape[0], 1, 1))
            self.poses = compose_transformations(trans_10, self.poses).numpy()

        # Adjust number of images according to strides
        self.n_img = len(self.color_paths)

    def switch_to_rgbd_gt(self):
        """When evaluating, we want to use the ground truth depth maps."""
        self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "depth_left", "*.png")))
        self.depth_paths = self.depth_paths[:: self.stride]

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = list(map(float, lines[i].split()))
            rotation_matrix = quaternion_to_matrix(torch.tensor(line[3:])).detach().numpy()
            c2w = np.eye(4)
            c2w[:3, :3] = rotation_matrix
            c2w[:3, 3] = line[:3]

            self.poses.append(c2w)


class DAVIS(BaseDataset):
    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(DAVIS, self).__init__(cfg, device)
        self.sequence = cfg.data.scene
        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, self.sequence, "*.jpg")))

        self.has_dyn_masks = True
        self.mask_path = self.input_folder.replace("JPEGImages", "Annotations")
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_path, self.sequence, "*.png")))

        # For Pseudo RGBD, we use monocular depth predictions in another folder
        if cfg.mode == "prgbd":
            self.depth_path = self.input_folder.replace(
                "JPEGImages/Full-Resolution", "Depth/Full-Resolution/" + self.mono_model
            )
            self.depth_paths = sorted(glob.glob(os.path.join(self.depth_path, self.sequence, "*.npy")))
            assert (
                len(self.depth_paths) == self.n_img
            ), f"Number of depth maps {len(self.depth_paths)} does not match number of images {self.n_img}"
        else:
            self.depth_paths = None

        if self.depth_paths is not None:
            if self.t_stop is not None:
                self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[:: self.stride]

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
            self.mask_paths = self.mask_paths[self.t_start : self.t_stop]
        self.color_paths = self.color_paths[:: self.stride]
        self.mask_paths = self.mask_paths[:: self.stride]
        self.n_img = len(self.color_paths)

        self.poses = None


class TotalRecon(BaseDataset):
    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(TotalRecon, self).__init__(cfg, device)

        self.input_folder = os.path.join(cfg.data.input_folder, cfg.data.scene + "-stereo000-leftcam")
        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, "images/*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(self.input_folder, "masks/*.png")))
        self.background_value = 0

        # Set number of images for loading poses
        self.n_img = len(self.color_paths)
        self.pose_paths = sorted(glob.glob(os.path.join(self.input_folder, "camera_rtks/*.txt")))

        # For Pseudo RGBD, we use monocular depth predictions in another folder
        if cfg.mode == "rgbd":
            self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "depths", "*.depth")))
            assert (
                len(self.depth_paths) == self.n_img
            ), f"Number of depth maps {len(self.depth_paths)} does not match number of images {self.n_img}"
        elif cfg.mode == "prgbd":
            self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, self.mono_model, "*.npy")))
            assert (
                len(self.depth_paths) == self.n_img
            ), f"Number of depth maps {len(self.depth_paths)} does not match number of images {self.n_img}"
        else:
            self.depth_paths = None

        if self.depth_paths is not None:
            if self.t_stop is not None:
                self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[:: self.stride]

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
            self.mask_paths = self.mask_paths[self.t_start : self.t_stop]
            self.pose_paths = self.pose_paths[self.t_start : self.t_stop]

        self.color_paths = self.color_paths[:: self.stride]
        self.mask_paths = self.mask_paths[:: self.stride]
        self.pose_paths = self.pose_paths[:: self.stride]

        self.has_dyn_masks = True

        self.load_poses(self.pose_paths)
        # self.set_intrinsics()

        # Set number of images for loading poses
        self.n_img = len(self.color_paths)

    def switch_to_rgbd_gt(self):
        """When evaluating, we want to use the ground truth depth maps."""
        self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "depths", "*.depth")))
        self.depth_paths = self.depth_paths[:: self.stride]

    def load_poses(self, paths):
        self.poses = []
        for path in paths:
            RTK = np.loadtxt(path)

            w2c = np.eye(4)
            w2c[:3, :3] = RTK[:3, :3]
            w2c[:3, 3] = RTK[:3, 3]
            c2w = np.linalg.inv(w2c)
            self.poses.append(c2w)

    def set_intrinsics(self):
        RTK = np.loadtxt(self.pose_paths[0])
        self.fx, self.fy, self.cx, self.cy = RTK[-1]


# NOTE this is the KITTI VO dataset
# we would need to build a different dataloader for the Depth or Sceneflow dataset
class KITTI(BaseDataset):
    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(KITTI, self).__init__(cfg, device)
        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, "image_2/*.png")))

        sequence = Path(self.input_folder).name
        # Set number of images for loading poses
        self.n_img = len(self.color_paths)
        # For Pseudo RGBD, we use monocular depth predictions in another folder
        if cfg.mode == "prgbd":
            self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "depth", self.mono_model, "*.npy")))
            assert (
                len(self.depth_paths) == self.n_img
            ), f"Number of depth maps {len(self.depth_paths)} does not match number of images {self.n_img}"
        else:
            self.depth_paths = None

        if self.depth_paths is not None:
            if self.t_stop is not None:
                self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[:: self.stride]

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
        self.color_paths = self.color_paths[:: self.stride]
        # Set number of images for loading poses
        self.n_img = len(self.color_paths)

        self.pose_path = os.path.join(self.input_folder, "../../poses", f"{sequence}.txt")
        # We only have groundtruth poses for sequences 00 to 10, i.e. 11 training sequences
        # The other sequences are part of the online test evaluation for the benchmark.
        if os.path.exists(self.pose_path):
            self.poses = self.load_poses(self.pose_path)
            if self.t_stop is not None:
                self.poses = self.poses[self.t_start : self.t_stop]
            self.poses = self.poses[:: self.stride]
        else:
            self.poses = None

    # FIXME poses need to be returned so that they are consistent with our evaluation code
    def load_poses(self, path: str):
        """KITTI stores 4x4 homogeneous matrices as 12 entry rows."""
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        for i in range(len(lines)):
            line = list(map(float, lines[i].split()))
            c2w = np.eye(4)
            c2w[:3, :] = np.array(line).reshape(3, 4)
            poses.append(c2w)

        return poses


class Azure(BaseDataset):
    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(Azure, self).__init__(cfg, device)
        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, "color", "*.jpg")))
        self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "depth", "*.png")))
        self.load_poses(os.path.join(self.input_folder, "scene", "trajectory.log"))

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.poses = self.poses[self.t_start : self.t_stop]

        self.color_paths = self.color_paths[:: self.stride]
        self.depth_paths = self.depth_paths[:: self.stride]
        self.poses = self.poses[:: self.stride]
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        if os.path.exists(path):
            with open(path) as f:
                content = f.readlines()

                # Load .log file.
                for i in range(0, len(content), 5):
                    # format %d (src) %d (tgt) %f (fitness)
                    data = list(map(float, content[i].strip().split(" ")))
                    ids = (int(data[0]), int(data[1]))
                    fitness = data[2]

                    # format %f x 16
                    c2w = np.array(list(map(float, ("".join(content[i + 1 : i + 5])).strip().split()))).reshape((4, 4))

                    self.poses.append(c2w)
        else:
            for i in range(self.n_img):
                c2w = np.eye(4)
                self.poses.append(c2w)


class ScanNet(BaseDataset):
    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(ScanNet, self).__init__(cfg, device)
        max_frames = cfg.get("max_frames", -1)
        if max_frames < 0:
            max_frames = int(1e5)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "color", "*.jpg")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )[:max_frames]
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "depth", "*.png")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )[:max_frames]
        self.load_poses(os.path.join(self.input_folder, "pose"))
        self.poses = self.poses[:max_frames]

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.poses = self.poses[self.t_start : self.t_stop]

        self.color_paths = self.color_paths[:: self.stride]
        self.depth_paths = self.depth_paths[:: self.stride]
        self.poses = self.poses[:: self.stride]

        self.n_img = len(self.color_paths)
        print("INFO: {} images got!".format(self.n_img))

    def switch_to_rgbd_gt(self):
        """When evaluating, we want to use the ground truth depth maps."""
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "depth_gt", "*.png")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )[:: self.stride]

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(
            glob.glob(os.path.join(path, "*.txt")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(" ")))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(TUM_RGBD, self).__init__(cfg, device)

        self.indices = None  # Since we have a different number of frames in rgbd and prgbd mode, we have to map this
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, cfg, frame_rate=32, mode=cfg.mode
        )

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.indices = self.indices[self.t_start : self.t_stop]
            self.poses = None if self.poses is None else self.poses[self.t_start : self.t_stop]

        self.cfg = cfg
        self.color_paths = self.color_paths[:: self.stride]
        self.depth_paths = self.depth_paths[:: self.stride]
        self.indices = self.indices[:: self.stride]
        self.poses = None if self.poses is None else self.poses[:: self.stride]
        self.n_img = len(self.color_paths)

    def switch_to_rgbd_gt(self):
        """We dont have the same number of depth and image frames. When we evaluate mono or prgbd, we
        will therefore end up with a different number of frames. Since we want to evaluate the geometry with a gt reference,
        we do a switch here to the gt depth maps.
        """
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, self.cfg, frame_rate=32, mode="rgbd"
        )
        self.color_paths = self.color_paths[:: self.stride]
        self.depth_paths = self.depth_paths[:: self.stride]
        self.indices = self.indices[:: self.stride]
        self.poses = None if self.poses is None else self.poses[:: self.stride]
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """read list data"""
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """pair images, depths, and poses"""
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath: str, cfg: DictConfig, frame_rate: int = -1, mode: str = "rgbd"):
        """read video data in tum-rgbd format"""
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        if mode == "prgbd":
            depth_paths = sorted(glob.glob(os.path.join(datapath, self.mono_model, "*.npy")))
            assert len(depth_paths) == len(
                image_data
            ), f"Number of depth maps {len(depth_paths)} does not match number of images {len(image_data)}"
            tstamp_depth = []
            for dpath in depth_paths:
                tstamp_depth.append(float(Path(dpath).stem))
            tstamp_depth = np.array(tstamp_depth, dtype=np.float64)
        else:
            depth_data = self.parse_list(depth_list).astype("<U512")  # FIX extended the dtype to include long strings
            tstamp_depth = depth_data[:, 0].astype(np.float64)

        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.indices = []  # Memoize which actual global frame we use for a given depth
        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]

            images += [os.path.join(datapath, image_data[i, 1])]
            if mode == "prgbd":
                depths += [os.path.join(datapath, depth_paths[j])]
            else:
                depths += [os.path.join(datapath, depth_data[j, 1])]
            self.indices.append(i)  # Memoize which actual frame we are using for a given depth
            # timestamp tx ty tz qx qy qz qw
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])

            # NOTE Transform trajectory, so first pose is always idenity
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w

            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """convert 4x4 pose matrix to (t, q)"""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


# TODO refactor to inherit from TUMRGBD, since it uses similar methods
class ETH3D(BaseDataset):
    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(ETH3D, self).__init__(cfg, device)
        (
            self.color_paths,
            self.depth_paths,
            self.poses,
            self.image_timestamps,
        ) = self.loadtum(self.input_folder, frame_rate=-1)

        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
            self.depth_paths = self.depth_paths[self.t_start : self.t_stop]
            self.poses = None if self.poses is None else self.poses[self.t_start : self.t_stop]
            self.image_timestamps = self.image_timestamps[self.t_start : self]

        self.color_paths = self.color_paths[:: self.stride]
        self.depth_paths = self.depth_paths[:: self.stride]
        self.poses = None if self.poses is None else self.poses[:: self.stride]
        self.image_timestamps = self.image_timestamps[:: self.stride]

        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """read list data"""
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """pair images, depths, and poses"""
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                # we need all images for benchmark, no max_dt checking here
                # if (np.abs(tstamp_depth[j] - t) < max_dt):
                associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """read video data in tum-rgbd format"""
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")
        else:
            pose_list = None

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)

        if pose_list is not None:
            pose_data = self.parse_list(pose_list, skiprows=1)
            pose_vecs = pose_data[:, 1:].astype(np.float64)

            tstamp_pose = pose_data[:, 0].astype(np.float64)
        else:
            tstamp_pose = None
            pose_vecs = None

        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        images, poses, depths, timestamps = [], [], [], tstamp_image
        if pose_list is not None:
            inv_pose = None
            for ix in range(len(associations)):
                (i, j, k) = associations[ix]
                images += [os.path.join(datapath, image_data[i, 1])]
                depths += [os.path.join(datapath, depth_data[j, 1])]
                # timestamp tx ty tz qx qy qz qw
                c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
                if inv_pose is None:
                    inv_pose = np.linalg.inv(c2w)
                    c2w = np.eye(4)
                else:
                    c2w = inv_pose @ c2w

                poses += [c2w]
        else:
            assert len(associations) == len(
                tstamp_image
            ), "Not all images are loaded. While benchmark need all images' pose!"
            print("\nDataset: no gt pose avaliable, {} images found\n".format(len(tstamp_image)))
            for ix in range(len(associations)):
                (i, j) = associations[ix]
                images += [os.path.join(datapath, image_data[i, 1])]
                depths += [os.path.join(datapath, depth_data[j, 1])]

            poses = None

        return images, depths, poses, timestamps

    def pose_matrix_from_quaternion(self, pvec):
        """convert 4x4 pose matrix to (t, q)"""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


class EuRoC(BaseDataset):
    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(EuRoC, self).__init__(cfg, device)
        self.color_paths, self.right_color_paths, self.poses = self.loadtum(self.input_folder, frame_rate=-1)
        if self.t_stop is not None:
            self.color_paths = self.color_paths[self.t_start : self.t_stop]
            self.right_color_paths = self.right_color_paths[self.t_start : self.t_stop]
            self.poses = None if self.poses is None else self.poses[self.t_start : self.t_stop]

        self.color_paths = self.color_paths[:: self.stride]
        self.right_color_paths = self.right_color_paths[:: self.stride]
        self.poses = None if self.poses is None else self.poses[:: self.stride]

        self.n_img = len(self.color_paths)

        K_l = np.array([458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0]).reshape(3, 3)
        d_l = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
        R_l = np.array(
            [
                0.999966347530033,
                -0.001422739138722922,
                0.008079580483432283,
                0.001365741834644127,
                0.9999741760894847,
                0.007055629199258132,
                -0.008089410156878961,
                -0.007044357138835809,
                0.9999424675829176,
            ]
        ).reshape(3, 3)

        P_l = np.array(
            [
                435.2046959714599,
                0,
                367.4517211914062,
                0,
                0,
                435.2046959714599,
                252.2008514404297,
                0,
                0,
                0,
                1,
                0,
            ]
        ).reshape(3, 4)
        map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3, :3], (752, 480), cv2.CV_32F)

        K_r = np.array([457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1]).reshape(3, 3)
        d_r = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]).reshape(5)
        R_r = np.array(
            [
                0.9999633526194376,
                -0.003625811871560086,
                0.007755443660172947,
                0.003680398547259526,
                0.9999684752771629,
                -0.007035845251224894,
                -0.007729688520722713,
                0.007064130529506649,
                0.999945173484644,
            ]
        ).reshape(3, 3)

        P_r = np.array(
            [
                435.2046959714599,
                0,
                367.4517211914062,
                -47.90639384423901,
                0,
                435.2046959714599,
                252.2008514404297,
                0,
                0,
                0,
                1,
                0,
            ]
        ).reshape(3, 4)
        map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3, :3], (752, 480), cv2.CV_32F)

        self.map_l = map_l
        self.map_r = map_r

    def load_left_image(self, path):
        img = cv2.remap(
            cv2.imread(path),
            self.map_l[0],
            self.map_l[1],
            interpolation=cv2.INTER_LINEAR,
        )

        return img

    def load_right_image(self, path):
        img = cv2.remap(
            cv2.imread(path),
            self.map_r[0],
            self.map_r[1],
            interpolation=cv2.INTER_LINEAR,
        )

        return img

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        color_data = self.load_left_image(color_path)

        H_out_with_edge, W_out_with_edge = (
            self.H_out + self.H_edge * 2,
            self.W_out + self.W_edge * 2,
        )

        color_data = cv2.resize(color_data, (W_out_with_edge, H_out_with_edge))
        color_data = (
            torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0
        )  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        if self.stereo:
            right_color_path = self.right_color_paths[index]
            right_color_data = self.load_right_image(right_color_path)
            right_color_data = cv2.resize(right_color_data, (W_out_with_edge, H_out_with_edge))
            right_color_data = (
                torch.from_numpy(right_color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0
            )  # bgr -> rgb, [0, 1]
            right_color_data = right_color_data.unsqueeze(dim=0)  # [1, 3, h, w]
            color_data = torch.cat([color_data, right_color_data], dim=0)

        depth_data = None
        intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
            intrinsic[2] -= edge

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
            intrinsic[3] -= edge

        if self.poses is not None:
            pose = matrix_to_lie(torch.tensor(self.poses[index])).float()
            ipdb.set_trace()
        else:
            pose = None

        return index, color_data, depth_data, intrinsic, pose

    def parse_list(self, filepath, skiprows=0):
        """read list data"""
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """pair images, depths, and poses"""
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            if tstamp_depth is None:
                k = np.argmin(np.abs(tstamp_pose - t))
                if np.abs(tstamp_pose[k] - t) < max_dt:
                    associations.append((i, k))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """read video data in tum-rgbd format"""
        # download from: https://github.com/princeton-vl/DROID-SLAM/tree/main/data/euroc_groundtruth
        scene_name = datapath.split("/")[-1]
        if os.path.isfile(os.path.join(datapath, f"{scene_name}.txt")):
            pose_list = os.path.join(datapath, f"{scene_name}.txt")
        else:
            raise ValueError(f"EuRoC_DATA_ROOT/{scene_name}/{scene_name}.txt doesn't exist!")

        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        image_list = sorted(glob.glob(os.path.join(datapath, "mav0/cam0/data/*.png")))
        right_image_list = [x.replace("cam0", "cam1") for x in image_list]
        tstamp_image = [float(img.split("/")[-1][:-4]) for img in image_list]
        tstamp_depth = None
        tstamp_pose = pose_data[:, 0].astype(np.float64)

        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        images, poses, right_images, intrinsics = [], [], [], []
        inv_pose = None
        for ix in range(len(associations)):
            (i, k) = associations[ix]
            images += [image_list[i]]
            right_images += [right_image_list[i]]
            # timestamp tx ty tz qx qy qz qw
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w

            poses += [c2w]

        return images, right_images, poses

    def pose_matrix_from_quaternion(self, pvec):
        """convert 4x4 pose matrix to (t, q)"""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


class Sintel(BaseDataset):
    def __init__(self, cfg: DictConfig, device: str = "cuda:0"):
        super(Sintel, self).__init__(cfg, device)

        self.has_dyn_masks = True
        self.background_value = 255

        # Check for endianness, based on Daniel Scharstein's optical flow code.
        # Using little-endian architecture, these two should be equal.
        self.tag_float = 202021.25
        self.tag_char = "PIEH"

        # NOTE chen: moving object masks are defined from frame pairs [t0, t1], we therefore only have n-1 in total
        # we therefore need to remove the last frame
        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, "final_left", cfg.data.scene, "*.png")))[
            :-1
        ]
        self.n_img = len(self.color_paths)
        self.mask_paths = sorted(glob.glob(os.path.join(self.input_folder, "rigidity", cfg.data.scene, "*.png")))

        # For Pseudo RGBD, we use monocular depth predictions in another folder
        if cfg.mode == "rgbd":
            self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "depth", cfg.data.scene, "*.dpt")))[
                :-1
            ]
            assert (
                len(self.depth_paths) == self.n_img
            ), f"Number of depth maps {len(self.depth_paths)} does not match number of images {self.n_img}"
            self.depth_paths = self.depth_paths[:: self.stride]
        elif cfg.mode == "prgbd":
            self.depth_paths = sorted(
                glob.glob(os.path.join(self.input_folder, self.mono_model, cfg.data.scene, "*.npy"))
            )[:-1]
            assert (
                len(self.depth_paths) == self.n_img
            ), f"Number of depth maps {len(self.depth_paths)} does not match number of images {self.n_img}"
            self.depth_paths = self.depth_paths[:: self.stride]
        else:
            self.depth_paths = None

        self.color_paths = self.color_paths[:: self.stride]
        self.mask_paths = self.mask_paths[:: self.stride]

        self.pose_paths = sorted(glob.glob(os.path.join(self.input_folder, "camdata_left", cfg.data.scene, "*.cam")))[
            :-1
        ]
        self.pose_paths = self.pose_paths[:: self.stride]

        self.load_poses(self.pose_paths)
        self.set_intrinsics()
        # Set number of images for loading poses
        self.n_img = len(self.color_paths)

    def switch_to_rgbd_gt(self):
        """When evaluating, we want to use the ground truth depth maps."""
        self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "results/depth*.png")))
        self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, "depth", cfg.data.scene, "*.dpt")))[:-1]
        self.depth_paths = self.depth_paths[:: self.stride]

    def load_poses(self, paths: List[str]):
        self.poses = []
        for path in paths:
            w2c = np.eye(4)
            K, T = self.cam_read(path)
            w2c[:3, :3] = T[:3, :3]
            w2c[:3, 3] = T[:, 3]
            c2w = np.linalg.inv(w2c)
            self.poses.append(c2w)

    def set_intrinsics(self):
        K, _ = self.cam_read(self.pose_paths[0])
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def cam_read(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read camera data, return (M,N) tuple.

        M is the intrinsic matrix, N is the extrinsic matrix, so that

        x = M*N*X,
        with x being a point in homogeneous image pixel coordinates, X being a
        point in homogeneous world coordinates.
        """
        f = open(filename, "rb")
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert (
            check == self.tag_float
        ), " cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
            self.tag_float, check
        )
        M = np.fromfile(f, dtype="float64", count=9).reshape((3, 3))
        N = np.fromfile(f, dtype="float64", count=12).reshape((3, 4))
        return M, N

    def flow_read(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read optical flow from file, return (U,V) tuple.

        Original code by Deqing Sun, adapted from Daniel Scharstein.
        """
        f = open(filename, "rb")
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert (
            check == self.tag_float
        ), " flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
            self.tag_float, check
        )
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert (
            width > 0 and height > 0 and size > 1 and size < 100000000
        ), " flow_read:: Wrong input size (width = {0}, height = {1}).".format(width, height)
        tmp = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width * 2))
        u = tmp[:, np.arange(width) * 2]
        v = tmp[:, np.arange(width) * 2 + 1]
        return u, v

    # NOTE since we have depth for the left camera, we dont really need the disparity
    def disparity_read(self, filename: str) -> np.ndarray:
        """Return disparity read from filename."""
        f_in = np.array(Image.open(filename))
        d_r = f_in[:, :, 0].astype("float64")
        d_g = f_in[:, :, 1].astype("float64")
        d_b = f_in[:, :, 2].astype("float64")

        depth = d_r * 4 + d_g / (2**6) + d_b / (2**14)
        return depth

    # NOTE this is for some super pixel segmentation, not the moving object masks
    def segmentation_read(filename: str) -> np.ndarray:
        """Return segmentation read from filename."""
        f_in = np.array(Image.open(filename))
        seg_r = f_in[:, :, 0].astype("int32")
        seg_g = f_in[:, :, 1].astype("int32")
        seg_b = f_in[:, :, 2].astype("int32")

        segmentation = (seg_r * 256 + seg_g) * 256 + seg_b
        return segmentation


dataset_dict = {
    "folder": ImageFolder,
    "replica": Replica,
    "scannet": ScanNet,
    "azure": Azure,
    "tumrgbd": TUM_RGBD,
    "eth3d": ETH3D,
    "euroc": EuRoC,
    "sintel": Sintel,
    "tartanair": TartanAir,
    "kitti": KITTI,
    "davis": DAVIS,
    "totalrecon": TotalRecon,
}
