from typing import Optional, Tuple, List, Dict
from termcolor import colored
import re
import gc
import time
from pathlib import Path
import ipdb
from omegaconf import DictConfig

import torch
import torchvision.transforms as tv
from torch.multiprocessing import Value

import cv2
import numpy as np

import lietorch
from .geom import projective_ops as pops
from .depth_video import DepthVideo
from .modules.corr import CorrBlock

import faiss
import faiss.contrib.torch_utils


def show_nan(tensor: torch.Tensor) -> None:
    """Plot a map of all nan's in a tensor"""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.imshow(tensor.isnan().squeeze().cpu().numpy())
    plt.axis("off")
    plt.show()


def merge_candidates(all_candidates: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """We communicate the loop candidates as Tuples (ii, jj) into a multiprocessing Queue.
    Since the loop detector runs extremely fast, the Queue will likely contain multiple sets when
    its being pulled from. We therefore might have to merge multiple together.
    """
    if len(all_candidates) == 1:
        return all_candidates[0]
    all_ii, all_jj = [], []
    for candidates in all_candidates:
        ii, jj = candidates
        all_ii.append(ii)
        all_jj.append(jj)
    all_ii, all_jj = torch.cat(all_ii), torch.cat(all_jj)
    return (all_ii, all_jj)


class LoopDetector:
    """This class is used to compute the Optical Flow between new frames and previous frames in the video and detect loops based on it.
    We simply return an exra set of edges to insert into a global factor graph.

    Some observations:
        - while our internal network is not as good as RAFT, it still captures the mean relative motion quite well.
        The range however is very limited, i.e. its much easier to filter correctly with RAFT, since very similar frames will have
        e.g. a distance of ~20-30, while most unsimilar frames have a distance > 50.0.

        - using a proper optical flow network like RAFT has much worse amortized costs. While a single
        batch pass with the small update network takes ~13ms, RAFT will eventually take 1-2s when we have to check over
        the whole history in later stages.

        - the internal update network does not produce good optical flow like RAFT, but has some correlation to it.
        We get at least a subset of the edges we would get with RAFT.

        - Since all loop detection methods have linear cost in the naive implementation, its impossible to get around. In order
        to keep costs low, you will have to use a low level descriptor and a database, which is the traidional way to do it.
        -> Place recognition based on image descriptor statistics and query a build up database.

        - Since LoopySLAM uses DBow3, we simply try this strategy as well and add edges to our factor graph based on it in backend.
    """

    def __init__(
        self,
        cfg: DictConfig,
        net: torch.nn.Module,
        video: DepthVideo,
        device: str = "cuda:0",
    ):
        self.cfg = cfg

        self.counter = Value("i", 0)
        self.video = video
        self.net = None  # see https://github.com/Lightning-AI/pytorch-lightning/issues/17637
        if self.video.cfg.mode == "stereo":
            raise NotImplementedError("Stereo mode not supported yet for loop closures!")
        imh, imw = self.video.ht, self.video.wd
        self.ht, self.wd = imh // self.video.scale_factor, imw // self.video.scale_factor

        self.method = self.cfg.method
        self.device = device
        # mean, std for image normalization
        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
        self.normalize = tv.Normalize(mean=self.MEAN, std=self.STDV)

        self.thresh = self.cfg.threshold

        ### Motion based similarity detectors
        if self.method == "raft":
            self.net, self.padder = self.load_raft(self.cfg.weights)  # Use actual RAFT optical flow network
        elif self.method == "internal":
            self.net = net  # Use the internal droid update network

        elif self.method == "eigen":
            ### Get the database
            # Flat index, i.e. we dont build up a large index over time (this would not amortize <10 000 queries)
            index_flat = faiss.IndexFlatL2(self.cfg.fc_output_dim)  # Measure L2 distance between features
            # see https://gist.github.com/mdouze/c7653aaa8c3549b28bad75bd67543d34 for how to setup max. cosine similarity
            if "cuda" in self.device:
                device_id = int(re.findall(r"\d+", self.device)[0])
            else:
                device_id = 0
            # NOTE chen: search on the GPU is pretty fast, but not pickable, see https://github.com/facebookresearch/faiss/issues/1306
            # -> You cannot use the indices in a multiprocessing setup, therefore use CPU!
            t_start = time.time()
            # self.db = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), device_id, index_flat)  # Put on GPU 0
            self.db = index_flat  # Use database on the CPU
            t_end = time.time()
            self.info(f"Initializing FAISS database on CPU took {t_end - t_start}s !")
            self.already_in_db, self.dbow_scores = [], []
        else:
            raise Exception(
                colored(
                    "Invalid loop detection method! Choose either 'raft', 'internal' or 'eigen'",
                    "red",
                )
            )

        # Frames/Nodes need to be at least n frames apart temporally
        self.min_temp_dist = self.cfg.min_temporal_distance
        # Loop closures should have a similar orientation
        self.max_orientation_diff = self.cfg.max_orientation_difference

        self.loop_candidates = []
        self.f_distances, self.rot_distances = {}, {}

    def info(self, msg) -> None:
        print(colored("[Loop Detection]: " + msg, "cyan"))

    # NOTE chen: make sure that RAFT is setup and in PYTHONPATH here
    # this is not standalone with our repo right now
    def load_raft(self, checkpoint: str) -> None:
        """Load a proper optical flow network like RAFT"""
        from easydict import EasyDict as edict
        import sys

        sys.path.append("ext/RAFT")
        from raft import RAFT
        from raft.utils.utils import InputPadder

        self.info(f"Loading RAFT model {self.cfg.weights}")
        model = torch.nn.DataParallel(RAFT(edict(self.cfg.model_cfg)))
        checkpoint = torch.load(checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint, strict=True)
        model = model.module
        model.eval().to(self.device)

        padder = InputPadder(self.video.images.shape)
        return model, padder

    def load_eigen(self):
        """Load the EigenPlaces model, see https://github.com/gmberton/EigenPlaces from ICCV23.
        This is a modern neural network for place recogntion.
        """
        model = torch.hub.load(
            "gmberton/eigenplaces", "get_trained_model", backbone=self.cfg.model, fc_output_dim=self.cfg.fc_output_dim
        )
        self.info(f"Loaded EigenPlaces model {self.cfg.model}!")
        return model.half().eval().to(self.device)

    @torch.no_grad()
    def get_eigen_features(self, idx: int) -> torch.Tensor:
        """Get EigenPlaces features for a specific frame index"""
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with self.video.get_lock():
                image = self.video.images[idx]
            image = self.normalize(image.unsqueeze(0) / 255.0)
            features = self.net(image)
        return features.cpu()

    def visualize_place_recognition_matches(
        self, query_idx: torch.Tensor, matches: Dict, show_only: Optional[int] = None
    ) -> None:
        """Visualize the place recognition matches with the query frame and the best matches."""
        import matplotlib.pyplot as plt

        matching_images = [self.video.images[idx].cpu().permute(1, 2, 0).numpy() for idx in matches.keys()]
        matching_ids = list(matches.keys())
        matching_scores = list(matches.values())

        if show_only is not None:
            num_plots = show_only + 1
        else:
            num_plots = len(matches) + 1
        fig, ax = plt.subplots(1, num_plots)
        ax[0].imshow(self.video.images[query_idx].cpu().permute(1, 2, 0).numpy())
        ax[0].set_title(f"Query [{query_idx}]")
        ax[0].axis("off")

        i = 0
        for idx, match_img, match_score in zip(matching_ids, matching_images, matching_scores):
            ax[i + 1].imshow(match_img)
            ax[i + 1].set_title(f"[{idx}]: {match_score:.2f}")
            ax[i + 1].axis("off")
            if show_only is not None:
                if i == show_only - 1:
                    break
            i += 1

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

    def compute_motion_loop(self, ii: torch.Tensor, jj: torch.Tensor, direction: str = "forward") -> torch.Tensor:
        """Compute optical flow between frames ii and jj in the video based on single examples.
        This works exactly the same as in motion_filter.

        We approximate flow magnitude using 1 update iteration
        NOTE chen: nets and inps are outputs of the context network, i.e. this is conditioned on the actual image ii
        """
        deltas = []
        for i, j in zip(ii, jj):
            fmap1, fmap2 = self.video.fmaps[i].to(self.device), self.video.fmaps[j].to(self.device)
            fmap1, fmap2 = fmap1.unsqueeze(0), fmap2.unsqueeze(0)
            ht, wd = self.video.ht // self.video.scale_factor, self.video.wd // self.video.scale_factor
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None, None]

            # Forward flow
            if direction == "forward":
                corr = CorrBlock(fmap2, fmap1)(coords0)
                _, delta, _ = self.net.update(self.video.nets[j][None, None], self.video.inps[j][None, None], corr)
            # Backward flow
            else:
                corr = CorrBlock(fmap1, fmap2)(coords0)
                _, delta, _ = self.net.update(self.video.nets[i][None, None], self.video.inps[i][None, None], corr)

            deltas.append(delta.squeeze())

        # Return [B, H // 8, W // 8 , 2] flow
        return torch.stack(deltas)

    def compute_motion_raft(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        iterations: int = 10,
        max_batch_size: int = 32,
        direction: str = "forward",
    ) -> torch.Tensor:
        """Compute motion by using a proper Optical Flow network like RAFT."""

        def inference(images1, images2, iterations, net, padder, direction="forward"):
            # RAFT expects images with 255 range
            if images1.max() <= 1.0:
                images1, images2 = images1, images2
            images1, images2 = padder.pad(images1, images2)  # Pads both images to same size

            # NOTE ii is here usually temporally ahead of jj, i.e. we compute the flow from jj to ii
            if direction == "forward":
                _, flow = net(images2, images1, iters=iterations, test_mode=True)
            else:
                _, flow = net(images1, images2, iters=iterations, test_mode=True)
            return flow

        # Chunk up into n batches with max_batch_size
        if len(ii) > max_batch_size:
            flow_i = []
            for i in range(0, len(ii), max_batch_size):
                ii_chunk, jj_chunk = ii[i : i + max_batch_size], jj[i : i + max_batch_size]
                images1, images2 = self.video.images[ii_chunk], self.video.images[jj_chunk]
                flow_i.append(inference(images1, images2, iterations, self.net, self.padder, direction=direction))
            flow = torch.cat(flow_i, dim=0)
        else:
            images1, images2 = self.video.images[ii], self.video.images[jj]
            flow = inference(images1, images2, iterations, self.net, self.padder, direction=direction)

        # Return [B, H, W, 2] flow
        return flow.permute(0, 2, 3, 1)

    def compute_motion_batch(self, ii: torch.Tensor, jj: torch.Tensor, direction: str = "forward") -> torch.Tensor:
        """Compute the optical flow between frame idx and all previous frames [0, idx-1].

        We approximate flow magnitude using 1 update iteration
        NOTE chen: nets and inps are outputs of the context network, i.e. this is conditioned on the actual image ii
        """
        fmap1, fmap2 = self.video.fmaps[ii].to(self.device), self.video.fmaps[jj].to(self.device)
        fmap1, fmap2 = fmap1[:, 0].unsqueeze(0), fmap2[:, 0].unsqueeze(0)  # [1, len(ii), 128, imh//8, imw//8]
        # [1, len(ii), imh//8, imw//8, 2]
        coords0 = pops.coords_grid(self.ht, self.wd, device=self.device)[None, None].repeat(1, len(ii), 1, 1, 1)

        # Forward flow
        if direction == "forward":
            corr = CorrBlock(fmap2, fmap1)(coords0)
            _, delta, _ = self.net.update(self.video.nets[jj].unsqueeze(0), self.video.inps[jj].unsqueeze(0), corr)
        # Backward flow
        else:
            corr = CorrBlock(fmap1, fmap2)(coords0)
            _, delta, _ = self.net.update(self.video.nets[ii].unsqueeze(0), self.video.inps[ii].unsqueeze(0), corr)

        # Returns [B, H // 8, W // 8, 2] flow
        return delta[0]

    def filter_duplicates(self, ii: torch.Tensor, jj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter out duplicate edges in the loop candidates"""
        combined = torch.cat((ii.unsqueeze(1), jj.unsqueeze(1)), dim=1)
        # Get unique elements along dim=0 (assuming each row represents a tuple)
        unique, indices = torch.unique(combined, return_inverse=True, dim=0)
        # Split the unique elements back into ii and jj
        ii_filtered, jj_filtered = unique[:, 0], unique[:, 1]
        return ii_filtered, jj_filtered

    def get_motion_distance(self, flow: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Compute weighted mean in valid regions.

        args:
        ---
        flow [torch.Tensor]: Optical Flow in shape [B, H, W, 2]
        valid [torch.Tensor]: Mask in shape [B, H, W]
        """
        bs = flow.shape[0]
        flow = torch.linalg.norm(flow, dim=-1)  # Reduce to [B, H, W]
        valid = valid.view(bs, -1)  # Reduce to [B, H*W]
        mean_of = (flow.view(bs, -1) * valid).sum(dim=-1) / valid.sum(dim=-1)
        # Return [B] means
        return mean_of

    def get_orientation_distance(self, ii, jj) -> torch.Tensor:
        """Compute the distance between the orientation of the current keyframe pose to all previous poses. We compute this
        for each edge (i,j) defined by the lists ii and jj.

        This assumes, that
        when optimizing highly similar loop closure candidates, the 3D position may not be the same due to drift, but the orientation
        will still be highly similar."""

        # Compute with SE3 directly
        g1, g2 = lietorch.SE3.InitFromVec(self.video.poses[ii]), lietorch.SE3.InitFromVec(self.video.poses[jj])
        g12 = g1 * g2.inv()
        d_se3 = g12.log()
        tau, phi = d_se3.split([3, 3], dim=-1)  # Separate into translation and rotation
        dt = tau.norm(dim=-1)  # NOTE We dont need the translation here
        dr = phi.norm(dim=-1)
        dr_deg = (180 / torch.pi) * dr  # convert radians to degrees

        return dr_deg

    def get_frame_distance(self, ii, jj, return_matches: bool = False):
        """Compute the visual difference between frames. This can be explicit motion or
        the distance between deep feature descriptors.
        """
        s = self.video.scale_factor

        # Get flow from i to all previous frames [0, i-1]
        if self.method == "raft":
            delta_i = self.compute_motion_raft(
                ii,
                jj,
                iterations=self.cfg.iterations,
                max_batch_size=self.cfg.max_batch_size,
                direction=self.cfg.direction,
            )
            valid = self.video.static_masks[ii]  # RAFT works on full resolution
            df = self.get_motion_distance(delta_i, valid)
            mask_df = df < self.thresh  # Candidates need to have a low optical flow difference

            # self.visualize_flow_2d(
            #     delta_i[0].permute(2, 0, 1),
            #     self.video.images[jj[0]].unsqueeze(0),
            #     self.video.images[ii[0]].unsqueeze(0),
            # )
            assert return_matches is False, "RAFT does not support place recognition!"

        elif self.method == "internal":
            delta_i = self.compute_motion_batch(ii, jj)
            with self.video.get_lock():
                valid = self.video.static_masks[ii, int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]

            df = self.get_motion_distance(delta_i, valid)
            mask_df = df < self.thresh  # Candidates need to have a low optical flow difference
            assert return_matches is False, "Internal DROID does not support place recognition!"

        elif self.method == "eigen":

            i = ii[0]
            # Compute appearance features and insert the frame into the database
            # Insert the first frame into the database
            if i == 1:
                feature_vec = self.get_eigen_features(0)
                self.db.add(feature_vec)  # add vectors to the database
                self.already_in_db.append(0)

            query_feature = self.get_eigen_features(i)
            # Search k nearest neighbors for query vector
            distances, matches = self.db.search(query_feature, min(self.db.ntotal, self.cfg.k_nearest))
            # NOTE chen: FAISS can only work on CPU in multi-threaded setup, so pull results back to GPU
            distances, matches = distances.to(self.device), matches.to(self.device)

            self.db.add(query_feature)  # add vectors to the database
            self.already_in_db.append(i)
            # Filter out invalid neighbor ids (will return -1 and 10**38 distance for invalid neighbors)
            valid = matches >= 0
            distances, neighbors = distances[valid], matches[valid]
            matches = {k: v for k, v in zip(neighbors, distances)}
            df = torch.inf * torch.ones(len(self.already_in_db) - 1, device=self.device)
            # Only set finite distance for valid matches
            df[neighbors] = distances
            mask_df = df < self.thresh  # Candidates need to have a small distance (high similarity score)

        else:
            raise Exception("Invalid loop detection method! Choose either 'raft', 'internal', 'bow' or 'eigen'.")

        if return_matches:
            return df, mask_df, matches
        else:
            return df, mask_df

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def check(self, insert_bidirectional: bool = True) -> Tuple[List]:
        """
        1. Motion based:
        ---
        Check if we have inserted new keyframes into our video datastructure.
        i) If yes, then use the memoized feature maps and image contexts to use the update network to compute optical flow.
        ii) We compute the optical flow between the latest frame and all previous frames in the video.
        iii) Memoize potential loop candidates between the latest frame and previous frames.
        We return such bidirectional edges, so we can insert them into the global factor graph for loop closures.

        Caveat:
        ---
        Normally we would have to run the update network multiple times and get a sequence of residual flows.
        We heuristically just run this once as a proxy for optical flow. Because this seems to be quite inprecise, we
        also make it possible to compute proper optical flow with an external network like RAFT.
        NOTE chen: I noticed that our network seems to get quite the good mean flow distance comparably, but the range is
        very hard to threshold. With RAFT you get proper distances like e.g. 20-30 for similar frames and > 50 for very unsimilar frames.

        HI-SLAM has an addititional orientation check, i.e. the current estimate/pose of the keyframe should be similar
        in orientation to previous keyframes, since drift more or less just changes the position, but not the orientation.

        2. Appearnace based
        ---
        Because Optical Flow may not be a good measure for loop detection on larger maps (there can still be a lot of motion between frames),
        we make use of more traditional place recognition techniques.
        i) ORB-SLAM uses a visual bag of words approach, based on ORB descriptor distributions.
        ii) This does not work well on a modern setup, so we use recent deep feature descriptors used in place recognition.
        """
        # NOTE chen: extract value here because it could change during this update in multi-thread setup
        with self.video.get_lock():
            kf_counter = self.video.counter.value

        # We need at least 2 frames in the video to compute motion
        if not self.counter.value < kf_counter or kf_counter < 2:
            return None

        start = time.time()
        candidates = []
        for i in range(max(self.counter.value - 1, 1), kf_counter - 1):
            # Get the latest frame and repeat index for all previous frames
            ii = torch.tensor(i, device=self.device).repeat(i)  # Repeat index for i-1 times
            jj = torch.arange(i, device=self.device)  # Get indices of all previous frames

            df, mask_df, matches = self.get_frame_distance(ii, jj, return_matches=True)
            dr = self.get_orientation_distance(ii, jj)  # NOTE this is returned in degrees
            dt = torch.abs(ii - jj)  # Temporal frame distance

            # Memoize these, for inspection later
            self.rot_distances[i], self.f_distances[i] = dr, df

            ### Threshold conditions
            mask_dt = dt > self.min_temp_dist  # Candidates should not be in a temporal neighborhood
            mask_dr = dr < self.max_orientation_diff  # Candidates need to have a similar orientation
            ii, jj = ii[mask_dt & mask_df & mask_dr], jj[mask_dt & mask_df & mask_dr]

            if len(ii) > 0:
                # self.visualize_place_recognition_matches(i, matches, show_only=self.cfg.k_nearest)
                # ipdb.set_trace()
                # Insert bidirectional edges
                if insert_bidirectional:
                    candidates.append((torch.cat((ii, jj)), torch.cat((jj, ii))))
                # Only insert a unidirectional edge
                else:
                    candidates.append(((ii), (jj)))

        # Increment to latest frame like video
        self.counter.value = kf_counter

        torch.cuda.empty_cache()
        gc.collect()

        end = time.time()
        elapsed_time = end - start

        if len(candidates) > 0:
            self.info(f"Loop detection took {elapsed_time:.2f}s")
            all_ii, all_jj = merge_candidates(candidates)
            unique_ii, unique_jj = self.filter_duplicates(all_ii, all_jj)
            self.info(
                f"Found {len(unique_ii)} loop candidates with edges: ({unique_ii.tolist()}) -> ({unique_jj.tolist()})!"
            )
            return (unique_ii.share_memory_(), unique_jj.share_memory_())
        else:
            return None
