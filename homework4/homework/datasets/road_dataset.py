from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

from . import road_transforms
from .road_utils import Track


def _frames_from_info(info) -> Dict[str, np.ndarray]:
    """Extract a frames dict from an np.load(info.npz) handle."""
    # Common keys used by transforms; keep anything present.
    wanted = {
        "front", "location", "distance_down_track",
        "P", "V", "waypoints", "speed", "angular_velocity"
    }
    frames = {}
    for k in info.files:
        if k in wanted:
            frames[k] = info[k]
    # Back-compat: some datasets pack everything under "frames" as an object array/dict
    if "frames" in info.files and isinstance(info["frames"].item(), dict):
        frames.update(info["frames"].item())
    return frames


class RoadDataset(Dataset):
    """
    SuperTux planning dataset.

    Supports two folder layouts:
      A) split-level metadata: <split>/info.npz
      B) per-episode metadata: <split>/<episode>/info.npz
    """

    def __init__(self, split_path: str, transform_pipeline: str = "default"):
        self.split_path = Path(split_path)

        # Detect episodes
        root_info = self.split_path / "info.npz"
        if root_info.exists():
            self.episode_dirs = [self.split_path]
        else:
            self.episode_dirs = sorted(
                d for d in self.split_path.iterdir()
                if d.is_dir() and (d / "info.npz").exists()
            )
            if not self.episode_dirs:
                raise FileNotFoundError(
                    f"No info.npz under '{self.split_path}' or any episode subfolder."
                )

        # Build per-episode caches: (frames, track)
        self.episode_frames: List[Dict[str, np.ndarray]] = []
        self.episode_track: List[Track] = []
        self.frame_index: List[Tuple[int, int]] = []  # (episode_idx, frame_idx)

        for ei, ep in enumerate(self.episode_dirs):
            info = np.load(ep / "info.npz", allow_pickle=True)

            # Build Track from metadata
            try:
                path_distance = info["path_distance"]
                path_nodes = info["path_nodes"]
                path_width = info["path_width"]
            except KeyError as e:
                raise KeyError(f"Missing key {e} in {ep/'info.npz'}") from e

            track = Track(path_distance=path_distance,
                          path_nodes=path_nodes,
                          path_width=path_width)
            self.episode_track.append(track)

            # Build frames dict
            frames = _frames_from_info(info)
            self.episode_frames.append(frames)

            # Determine number of frames
            if "location" in frames:
                num_frames = len(frames["location"])
            elif "num_frames" in info.files:
                num_frames = int(info["num_frames"])
            elif "frames" in info.files and hasattr(info["frames"].item(), "__len__"):
                num_frames = len(info["frames"].item())
            else:
                raise ValueError(f"Cannot infer number of frames for episode '{ep}'")

            # Index all frames for this episode
            for fi in range(num_frames):
                self.frame_index.append((ei, fi))

        # Build transform pipeline
        if transform_pipeline == "default":
            self.transform = road_transforms.Compose([
                road_transforms.ImageLoader(None),  # episode path set per-sample
                road_transforms.EgoTrackProcessor(track=self.episode_track[0]),
                road_transforms.NormalizeImage(),
                road_transforms.ToTensor(),
            ])
        elif transform_pipeline == "state_only":
            self.transform = road_transforms.Compose([
                road_transforms.EgoTrackProcessor(track=self.episode_track[0]),
                road_transforms.ToTensor(),
            ])
        else:
            raise ValueError(f"Unknown transform_pipeline: {transform_pipeline}")

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, idx: int) -> dict:
        ei, fi = self.frame_index[idx]
        episode_path = self.episode_dirs[ei]
        frames = self.episode_frames[ei]
        track = self.episode_track[ei]

        sample = {
            # keys expected by transforms
            "_idx": fi,
            "_frames": frames,
            "episode_path": str(episode_path),
        }

        # Dynamically set the episode_path for ImageLoader
        for t in getattr(self.transform, "transforms", []):
            if isinstance(t, road_transforms.ImageLoader):
                t.episode_path = Path(episode_path)
            # Update track for the current episode
            if isinstance(t, (road_transforms.EgoTrackProcessor,)):
                t.track = track

        sample = self.transform(sample)
        return sample
