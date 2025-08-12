# homework/datasets/road_dataset.py
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

from . import road_transforms
from .road_utils import Track


def _load_track_from_info(info) -> Track:
    """Build a Track from info['track'] dict with keys: path_nodes, path_distance, path_width."""
    if "track" not in info.files:
        raise KeyError("info.npz missing 'track'")
    tr_obj = info["track"]
    # allow_pickle=True -> usually an object array with a Python dict inside
    track_dict = tr_obj.item() if hasattr(tr_obj, "item") else tr_obj
    try:
        path_nodes = track_dict["path_nodes"]
        path_distance = track_dict["path_distance"]
        path_width = track_dict["path_width"]
    except KeyError as e:
        raise KeyError(f"track dict missing {e}") from e
    return Track(path_distance=path_distance, path_nodes=path_nodes, path_width=path_width)


def _load_frames_from_info(info, require_images: bool) -> Dict[str, np.ndarray]:
    """Extract frames dict from info['frames'] with at least the keys needed by the pipeline."""
    if "frames" not in info.files:
        raise KeyError("info.npz missing 'frames'")
    fr_obj = info["frames"]
    frames = fr_obj.item() if hasattr(fr_obj, "item") else fr_obj
    if not isinstance(frames, dict):
        raise TypeError(f"frames is not a dict: {type(frames)}")

    # Minimal keys
    needed = ["location"]
    if require_images:
        needed.append("front")
        needed.append("distance_down_track")  # used by EgoTrackProcessor for progress

    missing = [k for k in needed if k not in frames]
    if missing:
        raise KeyError(f"frames dict missing keys: {missing}")

    return frames


class RoadDataset(Dataset):
    """
    SuperTux planning dataset with per-episode metadata:
      <split>/<episode>/info.npz
    Also works if <split>/info.npz exists (treated as a single episode).
    """

    def __init__(self, split_path: str, transform_pipeline: str = "default"):
        self.split_path = Path(split_path)
        self.require_images = (transform_pipeline == "default")

        # Discover episodes
        if (self.split_path / "info.npz").exists():
            self.episode_dirs = [self.split_path]
        else:
            self.episode_dirs = sorted(
                d for d in self.split_path.iterdir()
                if d.is_dir() and (d / "info.npz").exists()
            )
            if not self.episode_dirs:
                raise FileNotFoundError(
                    f"No info.npz found in '{self.split_path}' or its subfolders."
                )

        # Load per-episode caches
        self.episode_tracks: List[Track] = []
        self.episode_frames: List[Dict[str, np.ndarray]] = []
        self.frame_index: List[Tuple[int, int]] = []

        for ei, ep in enumerate(self.episode_dirs):
            info = np.load(ep / "info.npz", allow_pickle=True)

            track = _load_track_from_info(info)
            frames = _load_frames_from_info(info, require_images=self.require_images)

            self.episode_tracks.append(track)
            self.episode_frames.append(frames)

            # Determine number of frames
            if "location" in frames:
                num_frames = len(frames["location"])
            elif "front" in frames:
                num_frames = len(frames["front"])
            else:
                raise ValueError(f"Cannot infer number of frames for episode '{ep}'")
            for fi in range(num_frames):
                self.frame_index.append((ei, fi))

        # Compose transform
        if transform_pipeline == "default":
            self.transform = road_transforms.Compose([
                road_transforms.ImageLoader(None),  # episode_path set per sample
                road_transforms.EgoTrackProcessor(self.episode_tracks[0]),
                road_transforms.NormalizeImage(),
                road_transforms.ToTensor(),
            ])
        elif transform_pipeline == "state_only":
            self.transform = road_transforms.Compose([
                road_transforms.EgoTrackProcessor(self.episode_tracks[0]),
                road_transforms.ToTensor(),
            ])
        else:
            raise ValueError(f"Unknown transform_pipeline: {transform_pipeline}")

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, idx: int) -> dict:
        ei, fi = self.frame_index[idx]
        ep = self.episode_dirs[ei]
        frames = self.episode_frames[ei]
        track = self.episode_tracks[ei]

        sample = {
            "_idx": fi,              # used by transforms to pick the frame
            "_frames": frames,       # avoid re-opening npz in transforms
            "episode_path": str(ep), # used by ImageLoader for image files
        }

        # Update per-sample episode + track in transforms
        for t in getattr(self.transform, "transforms", []):
            if isinstance(t, road_transforms.ImageLoader):
                t.episode_path = Path(ep)
            if isinstance(t, road_transforms.EgoTrackProcessor):
                t.track = track

        return self.transform(sample)
