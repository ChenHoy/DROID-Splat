from typing import Optional
import ipdb

import torch
import lietorch

from .geom import projective_ops as pops
from .depth_video import DepthVideo
from .modules.corr import CorrBlock


class MotionFilter:
    """This class is used to filter incoming frames and extract features"""

    def __init__(self, net: torch.nn.Module, video: DepthVideo, thresh: float = 2.5, device: str = "cuda:0"):
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image: torch.Tensor) -> torch.Tensor:
        """context features"""
        # image: [1, b, 3, h, w], net: [1, b, 128, h//8, w//8], inp: [1, b, 128, h//8, w//8]
        net, inp = self.cnet(image).split([128, 128], dim=2)
        return net.tanh().squeeze(dim=0), inp.relu().squeeze(dim=0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image: torch.Tensor) -> torch.Tensor:
        """feature for correlation volume"""
        # image: [1, b, 3, h, w], return: [1, b, 128, h//8, w//8]
        return self.fnet(image).squeeze(dim=0)

    def visualize_flow_2d(self, flow: torch.Tensor, image1: torch.Tensor, image2: torch.Tensor) -> None:
        """Sanity check to see if we actually compute the correct optical flow between two frames"""
        import matplotlib.pyplot as plt
        from .visualization import opticalflow2rgb

        flow_rgb = opticalflow2rgb(flow).squeeze()
        image1, image2 = (image1.permute(0, 2, 3, 1), image2.permute(0, 2, 3, 1))
        image1, image2 = image1.cpu().squeeze().numpy(), image2.cpu().squeeze().numpy()

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image1)
        ax[0].set_title("Frame 1")
        ax[0].axis("off")

        ax[1].imshow(image2)
        ax[1].set_title("Frame 2")
        ax[1].axis("off")

        ax[2].imshow(flow_rgb)
        ax[2].axis("off")
        ax[2].set_title("Optical Flow 1 -> 2")

        plt.show()

    def visualize_flow_1d(self, flow: torch.Tensor, image1: torch.Tensor, image2: torch.Tensor) -> None:
        """Sanity check to see if we actually compute the correct optical flow between two frames"""
        import matplotlib.pyplot as plt

        flow = torch.linalg.norm(flow, dim=-1).cpu().squeeze().numpy()
        image1, image2 = (image1.permute(0, 2, 3, 1), image2.permute(0, 2, 3, 1))
        image1, image2 = image1.cpu().squeeze().numpy(), image2.cpu().squeeze().numpy()

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image1)
        ax[0].set_title("Frame 1")
        ax[0].axis("off")

        ax[1].imshow(image2)
        ax[1].set_title("Frame 2")
        ax[1].axis("off")

        ax[2].imshow(flow, cmap="Spectral")
        ax[2].axis("off")
        ax[2].set_title("Optical Flow Mag.")

        plt.show()

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(
        self,
        timestamp,
        image: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
        gt_pose=None,
        static_mask=None,
    ):
        """main update operation - run on every frame in video"""

        scale_factor = self.video.scale_factor
        IdentityMat = lietorch.SE3.Identity(
            1,
        ).data.squeeze()

        batch, _, imh, imw = image.shape
        ht = imh // scale_factor
        wd = imw // scale_factor

        # normalize images, [b, 3, imh, imw] -> [1, b, 3, imh, imw], b=1 for mono, b=2 for stereo
        inputs = image.unsqueeze(dim=0).to(self.device)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)  # [b, c, imh//8, imw//8]

        ### always add first frame to the depth video ###
        left_idx = 0  # i.e., left image, for stereo case, we only store the hidden or input of left image
        if self.video.counter.value == 0:
            # [1, 128, imh//8, imw//8]
            net, inp = self.__context_encoder(
                inputs[
                    :,
                    [
                        left_idx,
                    ],
                ]
            )
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(
                timestamp,
                image[left_idx],
                IdentityMat,
                1.0,
                depth,
                intrinsic,
                gmap,
                net[left_idx],
                inp[left_idx],
                gt_pose,
                static_mask,
            )

        ### only add new frame if there is enough motion ###
        else:
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None, None]  # [1, 1, imh//8, imw//8, 2]
            # [1, 1, 4*49, imh//8, imw//8]
            corr = CorrBlock(self.fmap[None, [left_idx]], gmap[None, [left_idx]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)  # [1, 1, imh//8, imw//8, 2]

            if static_mask is not None:
                valid = static_mask[
                    None, None, int(scale_factor // 2 - 1) :: scale_factor, int(scale_factor // 2 - 1) :: scale_factor
                ]
            else:
                valid = torch.ones_like(delta[..., 0])

            # check motion magnitude / add new frame to video
            if delta.norm(dim=-1)[valid.bool()].mean().item() > self.thresh:

                last_idx = self.video.counter.value - 1
                # self.visualize_flow_2d(
                #     delta[0].permute(0, 3, 1, 2),
                #     image[left_idx].unsqueeze(0),
                #     self.video.images[last_idx].unsqueeze(0),
                # )

                self.count = 0
                net, inp = self.__context_encoder(inputs[:, [left_idx]])  # [1, 128, imh//8, imw//8]
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(
                    timestamp,
                    image[left_idx],
                    None,
                    None,
                    depth,
                    intrinsic,  # NOTE chen: we now scale intrinsics inside the video set_item method!
                    gmap,
                    net[left_idx],
                    inp[left_idx],
                    gt_pose,
                    static_mask,
                )

            else:
                self.count += 1
