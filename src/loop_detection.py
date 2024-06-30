from typing import Optional, Tuple, List, Dict
from termcolor import colored
import time
from pathlib import Path
import ipdb
from omegaconf import DictConfig

import torch
from torch.multiprocessing import Value

import cv2
import numpy as np

import lietorch
from .geom import projective_ops as pops
from .depth_video import DepthVideo
from .modules.corr import CorrBlock

import pydbow3 as bow


def show_nan(tensor: torch.Tensor) -> None:
    """Plot a map of all nan's in a tensor"""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.imshow(tensor.isnan().squeeze().cpu().numpy())
    plt.axis("off")
    plt.show()


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
        if self.video.cfg.mode == "stereo":
            raise NotImplementedError("Stereo mode not supported yet for loop closures!")
        imh, imw = self.video.ht, self.video.wd
        self.ht, self.wd = imh // self.video.scale_factor, imw // self.video.scale_factor

        self.method = self.cfg.method
        self.device = device
        # mean, std for image normalization
        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

        # Only consider edges, where the flow magnitude is smaller than this threshold
        # i.e. the frames look visually similar
        if self.method == "raft":
            self.thresh = self.cfg.threshold
            self.net, self.padder = self.load_raft(self.cfg.weights)
        elif self.method == "internal":
            self.thresh = self.cfg.threshold
            self.net = net
        # Use a proper visual similarity method without motion computation
        else:
            self.db = self.load_dbow(voc_path=self.cfg.vocabulary)
            self.orb = cv2.ORB_create()
            self.thresh = self.cfg.threshold
            # TODO how does the database storage work, can we reliably just keep everything in self.db?
            # Store orb features for each keyframe
            # TODO maybe delete this afterwards
            self.already_in_db, self.dbow_scores = [], []

        # Frames/Nodes need to be at least n frames apart temporally
        self.min_temp_dist = self.cfg.min_temporal_distance
        # Loop closures should have a similar orientation
        self.max_orientation_diff = self.cfg.max_orientation_difference

        self.loop_candidates = []
        self.of_distances, self.rot_distances = {}, {}

    def info(self, msg) -> None:
        print(colored("[Loop Detection]: " + msg, "cyan"))

    def load_raft(self, checkpoint: str) -> None:
        from easydict import EasyDict as edict
        import sys

        sys.path.append("ext/RAFT")
        from raft import RAFT
        from raft.utils.utils import InputPadder

        model = torch.nn.DataParallel(RAFT(edict(self.cfg.model_cfg)))
        checkpoint = torch.load(checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint, strict=True)
        model = model.module
        model.eval().to(self.device)

        padder = InputPadder(self.video.images.shape)
        return model, padder

    def load_dbow(self, voc_path: str = "DBoW3/orbvoc.dbow3") -> None:
        """Load a preexisting DBow3 vocabulary and database to use for loop detection."""
        voc = bow.Vocabulary()
        self.info(f"[Loop detector] Loading vocabulary from {voc_path}")
        voc.load(voc_path)
        db = bow.Database()
        db.setVocabulary(voc)
        print("\n")
        return db

    def extract_orb_features(self, img: torch.Tensor) -> torch.Tensor:
        """Compute ORB features with opencv, this requires going back and forth between numpy and torch"""
        img = np.float32(img.cpu().numpy() * 255)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(np.uint8(gray_img), None)
        # NOTE you can visualize the keypoints with:
        # img_with_kp = cv2.drawKeypoints(np.uint8(gray_img), kp, None, color=(0, 255, 0), flags=0)
        return des

    def get_orb_features(self, idx: int) -> torch.Tensor:
        """Get ORB features for a specific frame index"""
        image = self.video.images[idx]
        return self.extract_orb_features(image.permute(1, 2, 0))

    def insert_frame_into_dbow(self, idx: int) -> None:
        """Compute the ORB features for the current keyframe and insert them into the database. We return the
        computed features for querying later.
        """
        image = self.video.images[idx]
        features = self.extract_orb_features(image.permute(1, 2, 0))
        self.db.add(features)
        self.already_in_db.append(idx)
        return features

    def query_dbow(self, query_features, k_neighbors: int = 2) -> None | Dict:
        """Make a nearest neighbor lookup with query frame. We return the k best matches and the scores.

        NOTE: if you simply want to compute all matches, you can use the query method with k_neighbors=len(self.db) - 1
        """

        results = self.db.query(query_features, k_neighbors)
        matches = {result.Id: result.Score for result in results}
        if len(matches) > 0:
            self.dbow_scores.append(min(matches.values()))
            return matches
        else:
            self.dbow_scores.append(-1)
            return None

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
                images1, images2 = 255 * images1, 255 * images2
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

        def test_with_quat_directly(g1: lietorch.SE3, g2: lietorch.SE3) -> float:
            # Get rotation, translation
            t1, q1 = g1.vec().split([3, 4], -1)
            t2, q2 = g2.vec().split([3, 4], -1)

            r1, r2 = lietorch.SO3.InitFromVec(q1), lietorch.SO3.InitFromVec(q2)
            dR = r1 * r2.inv()
            ang = dR.log().norm(dim=-1)
            deg = (180 / torch.pi) * ang  # convert radians to degrees
            return deg

        # Compute with SE3 directly
        g1, g2 = lietorch.SE3.InitFromVec(self.video.poses[ii]), lietorch.SE3.InitFromVec(self.video.poses[jj])
        g12 = g1 * g2.inv()
        d_se3 = g12.log()
        tau, phi = d_se3.split([3, 3], dim=-1)  # Separate into translation and rotation
        dt = tau.norm(dim=-1)  # NOTE We dont need the translation here
        dr = phi.norm(dim=-1)
        dr_deg = (180 / torch.pi) * dr  # convert radians to degrees

        return dr_deg

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def check(self) -> Tuple[List]:
        """Check if we have inserted new keyframes into our video datastructure. If yes, then use
        the memoized feature maps and image contexts to use the update network to compute optical flow.
        We compute the optical flow between the latest frame and all previous frames in the video.
        We can then memoize potential loop candidates between the latest frame and previous frames.
        We return such bidirectional edges, so we can insert them into the global factor graph for loop closures.

        Normally we would have to run the update network multiple times and get a sequence of residual flows.
        We heuristically just run this once as a proxy for optical flow. Because this seems to be quite inprecise, we
        also make it possible to compute proper optical flow with an external network like RAFT.
        NOTE chen: I noticed that our network seems to get quite the good mean flow distance comparably, but the range is
        very hard to threshold. With RAFT you get proper distances like e.g. 20-30 for similar frames and > 50 for unsimilar frames.

        HI-SLAM has an addititional orientation check, i.e. the current estimate/pose of the keyframe should be similar
        in orientation to previous keyframes, since drift more or less just changes the position, but not the orientation.
        """
        # NOTE chen: extract value here because it could change during this update in multi-thread setup
        kf_counter = self.video.counter.value
        # We need at least 2 frames in the video to compute motion
        if not self.counter.value < kf_counter or kf_counter < 2:
            return

        s = self.video.scale_factor
        start = time.time()
        candidates = []
        for i in range(max(self.counter.value - 1, 1), self.video.counter.value - 1):
            # Get the latest frame and repeat index for all previous frames
            ii = torch.tensor(i, device=self.device).repeat(i)  # Repeat index for i-1 times
            jj = torch.arange(i, device=self.device)  # Get indices of all previous frames

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
            elif self.method == "internal":
                delta_i = self.compute_motion_batch(ii, jj)
                valid = self.video.static_masks[ii, int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
                df = self.get_motion_distance(delta_i, valid)
                mask_df = df < self.thresh  # Candidates need to have a low optical flow difference
            else:
                # Compute appearance features and insert the frame into the database
                # Insert the first frame into the database
                if i == 1:
                    features = self.get_orb_features(0)
                    self.db.add(features)
                    self.already_in_db.append(0)

                features = self.get_orb_features(i)
                matches = self.query_dbow(features, k_neighbors=self.cfg.k_nearest)
                self.db.add(features)
                self.already_in_db.append(i)

                # NOTE chen: Query may not return all previous frames, i.e. we need to set infinite distance for all
                df = -torch.inf * torch.ones(len(self.already_in_db) - 1, device=self.device)
                if matches is not None:
                    valid_matches = torch.tensor(list(matches.keys()), device=self.device)
                    valid_scores = torch.tensor(list(matches.values()), device=self.device)
                    # Only set finite distance for valid matches
                    df[valid_matches] = valid_scores
                mask_df = df > self.thresh  # Candidates need to have a high similarity score

            # self.visualize_flow_2d(
            #     delta_i[0].permute(2, 0, 1),
            #     self.video.images[jj[0]].unsqueeze(0),
            #     self.video.images[ii[0]].unsqueeze(0),
            # )

            dr = self.get_orientation_distance(ii, jj)  # NOTE this is returned in degrees
            dt = torch.abs(ii - jj)  # Temporal frame distance

            # TODO delete this after debugging! -> Determine good thresholds
            # Memoize these, for inspection later
            self.rot_distances[i] = dr
            self.of_distances[i] = df

            ### Threshold conditions
            mask_dt = dt > self.min_temp_dist  # Candidates should not be in a temporal neighborhood
            mask_dr = dr < self.max_orientation_diff  # Candidates need to have a similar orientation
            ii, jj = ii[mask_dt & mask_df & mask_dr], jj[mask_dt & mask_df & mask_dr]

            if len(ii) > 0:
                ipdb.set_trace()
                self.visualize_place_recognition_matches(i, matches, show_only=self.cfg.k_nearest)
                # Insert bidirectional edges
                candidates.append((torch.cat((ii, jj)), torch.cat((jj, ii))))

        # Increment to latest frame like video
        self.counter.value = kf_counter

        end = time.time()
        elapsed_time = end - start
        self.info(f"Loop detection took {elapsed_time:.2f}s")

        if len(candidates) > 0:
            all_ii, all_jj = [], []
            for ii, jj in candidates:
                all_ii.append(ii)
                all_jj.append(jj)
            all_ii, all_jj = torch.cat(all_ii), torch.cat(all_jj)
            unique_ii, unique_jj = self.filter_duplicates(all_ii, all_jj)
            self.info(
                f"Found {len(unique_ii)} loop candidates with edges: ({unique_ii.tolist()}) -> ({unique_jj.tolist()})!"
            )
            return (unique_ii, unique_jj)
        else:
            return
