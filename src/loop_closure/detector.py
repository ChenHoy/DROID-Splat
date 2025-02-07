import ipdb
from typing import Optional, Tuple, List, Dict
from termcolor import colored
import re
import gc
import time
from omegaconf import DictConfig

import torch
import torchvision.transforms as tv
from torch.multiprocessing import Value

import lietorch
from ..depth_video import DepthVideo

import faiss
import faiss.contrib.torch_utils


def show_nan(tensor: torch.Tensor) -> None:
    """Plot a map of all nan's in a tensor"""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.imshow(tensor.isnan().squeeze().cpu().numpy())
    plt.axis("off")
    plt.show()


def merge_candidates(
    all_candidates: List[Tuple[torch.Tensor, torch.Tensor]], scores: Optional[List[torch.Tensor]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """We communicate the loop candidates as Tuples (ii, jj) into a multiprocessing Queue.
    Since the loop detector runs extremely fast, the Queue will likely contain multiple sets when
    its being pulled from. We therefore might have to merge multiple together.
    """
    if len(all_candidates) == 1:
        if scores is not None:
            return all_candidates[0][0], all_candidates[0][1], scores[0]
        else:
            return all_candidates[0][0], all_candidates[0][1]

    all_ii, all_jj, all_scores = [], [], []
    for i, candidates in enumerate(all_candidates):
        all_ii.append(candidates[0])
        all_jj.append(candidates[1])
        if scores is not None:
            all_scores.append(scores[i])

    all_ii, all_jj = torch.cat(all_ii), torch.cat(all_jj)
    if scores is not None:
        scores = torch.cat(scores)
        return all_ii, all_jj, scores
    else:
        return all_ii, all_jj


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

    def __init__(self, cfg: DictConfig, video: DepthVideo, device: str = "cuda:0"):
        self.cfg = cfg

        self.counter = Value("i", 0)
        self.video = video
        self.net = None  # see https://github.com/Lightning-AI/pytorch-lightning/issues/17637
        if self.video.cfg.mode == "stereo":
            raise NotImplementedError("Stereo mode not supported yet for loop closures!")
        imh, imw = self.video.ht, self.video.wd
        self.ht, self.wd = imh // self.video.scale_factor, imw // self.video.scale_factor

        self.device = device
        # mean, std for image normalization
        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
        self.normalize = tv.Normalize(mean=self.MEAN, std=self.STDV)

        self.thresh = self.cfg.threshold

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

        # Frames/Nodes need to be at least n frames apart temporally
        self.min_temp_dist = self.cfg.min_temporal_distance
        # Loop closures should have a similar orientation
        self.max_orientation_diff = self.cfg.max_orientation_difference

        self.f_distances, self.rot_distances = {}, {}

    def info(self, msg) -> None:
        print(colored("[Loop Detection]: " + msg, "cyan"))

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
            image = self.normalize(image.unsqueeze(0))
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

    def filter_duplicates(
        self, ii: torch.Tensor, jj: torch.Tensor, scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter out duplicate edges in the loop candidates"""
        combined = torch.cat((ii.unsqueeze(1), jj.unsqueeze(1)), dim=1)
        # Get unique elements along dim=0 (assuming each row represents a tuple)
        unique, indices = torch.unique(combined, return_inverse=True, dim=0)
        # We might have multiple duplicates with different scores -> take best score
        unique2score = {}
        for idx in torch.unique(indices):
            unique2score[combined[idx]] = torch.max(scores[indices == idx])

        # Split the unique elements back into ii and jj
        ii_filtered, jj_filtered = unique[:, 0], unique[:, 1]
        if scores is not None:
            if scores.ndim > 1:
                for idx, (i, j) in enumerate(zip(ii_filtered, jj_filtered)):
                    scores[idx] = unique2score[(i, j)]
            return ii_filtered, jj_filtered, scores
        else:
            return ii_filtered, jj_filtered, scores

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

        if return_matches:
            return df, mask_df, matches
        else:
            return df, mask_df

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def __call__(self, insert_bidirectional: bool = False) -> Tuple[List]:
        """
        Appearance based loop check
        ---
        Because Optical Flow may not be a good measure for loop detection on larger maps (there can still be a lot of motion between frames),
        we make use of more traditional place recognition techniques.
        i) ORB-SLAM uses a visual bag of words approach, based on ORB descriptor distributions.
        ii) This does not work well on a modern setup, so we use recent deep feature descriptors used in place recognition.

        HI-SLAM has an addititional orientation check, i.e. the current estimate/pose of the keyframe should be similar
        in orientation to previous keyframes, since drift more or less just changes the position, but not the orientation.

        """
        # NOTE chen: extract value here because it could change during this update in multi-thread setup
        with self.video.get_lock():
            kf_counter = self.video.counter.value

        # We need at least 2 frames in the video to compute motion
        if not self.counter.value < kf_counter or kf_counter < 2:
            return None

        start = time.time()
        candidates, scores = [], []
        for i in range(max(self.counter.value - 1, 1), kf_counter - 1):
            # Get the latest frame and repeat index for all previous frames
            ii = torch.tensor(i, device=self.device).repeat(i)  # Repeat index for i-1 times
            jj = torch.arange(i, device=self.device)  # Get indices of all previous frames

            df, mask_df, matches = self.get_frame_distance(ii, jj, return_matches=True)
            dr = self.get_orientation_distance(ii, jj)  # NOTE this is returned in degrees
            dt = torch.abs(ii - jj)  # Temporal frame distance

            # Memoize these, for maybe later inspection
            self.rot_distances[i], self.f_distances[i] = dr, df

            ### Threshold conditions
            mask_dt = dt > self.min_temp_dist  # Candidates should be at least n frames apart
            mask_dr = dr < self.max_orientation_diff  # Candidates need to have a similar orientation
            ii, jj = ii[mask_dt & mask_df & mask_dr], jj[mask_dt & mask_df & mask_dr]
            score = df[mask_dt & mask_df & mask_dr]

            if len(ii) > 0:
                # self.visualize_place_recognition_matches(i, matches, show_only=self.cfg.k_nearest)
                # Insert bidirectional edges
                if insert_bidirectional:
                    candidates.append((torch.cat((ii, jj)), torch.cat((jj, ii))))
                    scores.append(torch.cat((score, score)))
                # Only insert a unidirectional edge
                else:
                    candidates.append(((ii), (jj)))
                    scores.append(score)

        # Increment to latest frame like video
        self.counter.value = kf_counter

        torch.cuda.empty_cache()
        gc.collect()

        end = time.time()
        elapsed_time = end - start

        if len(candidates) > 0:

            self.info(f"Loop detection took {elapsed_time:.2f}s")
            all_ii, all_jj, scores = merge_candidates(candidates, scores)
            unique_ii, unique_jj, scores = self.filter_duplicates(all_ii, all_jj, scores)
            self.info(
                f"Found {len(unique_ii)} loop candidates with edges: ({unique_ii.tolist()}) -> ({unique_jj.tolist()})!"
            )
            return unique_ii, unique_jj, scores
        else:
            return None
