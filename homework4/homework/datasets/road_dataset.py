# homework/datasets/road_dataset.py
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

from . import road_transforms
from .road_utils import Track


def _to_2d(a, last_dim=None) -> np.ndarray:
    """
    Ensure 'a' is a real 2-D float32 ndarray.
    Handles arrays of objects (array of vectors) by stacking.
    If last_dim is given, assert that a.shape[1] == last_dim (when non-empty).
    """
    x = np.asarray(a)
    # If it's an object array of vectors, stack them
    if x.dtype == object or (x.ndim == 1 and x.size and isinstance(x[0], (list, tuple, np.ndarray))):
        x = np.stack([np.asarray(v, dtype=np.float32) for v in x], axis=0)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        # a single vector -> make it (1, D)
        x = x[None, :]
    if last_dim is not None and x.size > 0:
        assert x.shape[1] == last_dim, f"Expected last dim {last_dim}, got {x.shape}"
    return x


def _load_track_from_info(info) -> Track:
    """Build a Track from info['track'] dict with keys: path_nodes, path_distance, path_width."""
    if "track" not in info.files:
        raise KeyError("info.npz missing 'track'")
    tr_obj = info["track"]
    track_dict = tr_obj.item() if hasattr(tr_obj, "item") else tr_obj
    path_nodes = track_dict["path_nodes"]
    path_distance = track_dict["path_distance"]
    path_width = track_dict["path_width"]
    return Track(path_distance=path_distance, path_nodes=path_nodes, path_width=path_width)


def _load_frames_from_info(info) -> Dict[str, np.ndarray]:
    """Extract frames dict from info['frames'] and coerce arrays to sane shapes."""
    if "frames" not in info.files:
        raise KeyError("info.npz missing 'frames'")
    fr_obj = info["frames"]
    frames = fr_obj.item() if hasattr(fr_obj, "item") else fr_obj
    if not isinstance(frames, dict):
        raise TypeError(f"frames is not a dict: {type(frames)}")

    # Required by EgoTrackProcessor: front, location, distance_down_track
    if "location" not in frames or "front" not in frames:
        raise KeyError("frames dict must contain 'front' and 'location'")

    # Force consistent dtypes/shapes
    frames["location"] = _to_2d(frames["location"], last_dim=3)          # (N,3)
    frames["front"] = _to_2d(frames["front"], last_dim=3)                # (N,3)
    if "distance_down_track" in frames:
        dd = np.asarray(frames["distance_down_track"]).reshape(-1).astype(np.float32)  # (N,)
        frames["distance_down_track"] = dd
    else:
        # Minimal fallback: 0..N-1 spacing if not present
        frames["distance_down_track"] = np.arange(len(frames["location"]), dtype=np.float32)

    return frames


class RoadDataset(Dataset):
    """
    SuperTux planning dataset.

    Supports both:
      - split-level metadata: <split>/info.npz
      - per-episode metadata: <split>/<episode>/info.npz
    """

    def __init__(self, split_path: str, transform_pipeline: str = "default"):
        self.split_path = Path(split_path)

        # Find episodes
        if (self.split_path / "info.npz").exists():
            self.episode_dirs = [self.split_path]
        else:
            self.episode_dirs = sorted(
                d for d in self.split_path.iterdir() if d.is_dir() and (d / "info.npz").exists()
            )
            if not self.episode_dirs:
                raise FileNotFoundError(f"No info.npz found in '{self.split_path}' or its subfolders.")

        # Load per-episode caches
        self.episode_tracks: List[Track] = []
        self.episode_frames: List[Dict[str, np.ndarray]] = []
        self.frame_index: List[Tuple[int, int]] = []

        for ei, ep in enumerate(self.episode_dirs):
            info = np.load(ep / "info.npz", allow_pickle=True)
            track = _load_track_from_info(info)
            frames = _load_frames_from_info(info)

            self.episode_tracks.append(track)
            self.episode_frames.append(frames)

            # Frame count
            num_frames = len(frames["location"])
            for fi in range(num_frames):
                self.frame_index.append((ei, fi))

        # Compose transform
        if transform_pipeline == "default":
            self.transform = road_transforms.Compose([
                road_transforms.ImageLoader(self.episode_dirs[0]),
                road_transforms.EgoTrackProcessor(self.episode_tracks[0]),
                road_transforms.ToTensor(),       # tensor first
                road_transforms.NormalizeImage(), # then normalize
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
            "_idx": fi,                # used by transforms
            "_frames": frames,         # avoid re-opening npz
            "episode_path": str(ep),   # used by ImageLoader
        }

        # Point transforms to the current episode + track
        for t in getattr(self.transform, "transforms", []):
            if isinstance(t, road_transforms.ImageLoader):
                t.episode_path = Path(episode_path)
            if isinstance(t, road_transforms.EgoTrackProcessor):
                t.track = track


        return self.transform(sample)
