from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

from . import road_transforms
from .road_utils import Track


def _frames_from_info(info) -> Dict[str, np.ndarray]:
    """
    Extract frame arrays (as numpy arrays) required by transforms.
    Works for both 'flat' npz and npz-within-dict formats.
    """
    frames = {}

    # Preferred top-level arrays
    for k in ("front", "location", "distance_down_track"):
        if k in info.files:
            frames[k] = info[k]

    # Some datasets pack everything under "frames" as an object dict
    if "frames" in info.files:
        try:
            maybe = info["frames"].item()
            if isinstance(maybe, dict):
                for k in ("front", "location", "distance_down_track"):
                    if k not in frames and k in maybe:
                        frames[k] = maybe[k]
        except Exception:
            pass

    # Minimal sanity
    missing = [k for k in ("front", "location", "distance_down_track") if k not in frames]
    if missing:
        raise KeyError(f"Missing frame keys {missing} in info.npz")

    return frames


def _track_from_any(ep: Path) -> Track:
    """
    Build a Track from any available representation in the episode folder.
    Priority:
      1) info.npz with path_distance/path_nodes/path_width
      2) track.npz (common alt)
      3) info.npz with center/width  -> synthesize path_nodes/path_distance
    """
    info = np.load(ep / "info.npz", allow_pickle=True)

    # 1) Full keys present
    if all(k in info.files for k in ("path_distance", "path_nodes", "path_width")):
        return Track(
            path_distance=info["path_distance"],
            path_nodes=info["path_nodes"],
            path_width=info["path_width"],
        )

    # 2) Separate track.npz
    track_npz = ep / "track.npz"
    if track_npz.exists():
        t = np.load(track_npz, allow_pickle=True)
        if all(k in t.files for k in ("path_distance", "path_nodes", "path_width")):
            return Track(
                path_distance=t["path_distance"],
                path_nodes=t["path_nodes"],
                path_width=t["path_width"],
            )
        # Accept center+width in track.npz too
        if "center" in t.files and "width" in t.files:
            center = np.float32(t["center"])
            width = np.float32(t["width"]).reshape(-1, 1)
            # Synthesize minimal path_nodes/path_distance
            path_nodes = np.stack([center, center], axis=1)  # (n,2,3)
            # Per-node segment lengths; 2nd col as "delta"
            center_delta = np.linalg.norm(np.diff(center, axis=0, prepend=center[:1]), axis=1)
            path_distance = np.stack([np.zeros_like(center_delta), center_delta], axis=1)  # (n,2)
            return Track(path_distance=path_distance, path_nodes=path_nodes, path_width=width)

    # 3) Fallback: info.npz with center+width
    if "center" in info.files and "width" in info.files:
        center = np.float32(info["center"])
        width = np.float32(info["width"]).reshape(-1, 1)
        path_nodes = np.stack([center, center], axis=1)  # (n,2,3)
        center_delta = np.linalg.norm(np.diff(center, axis=0, prepend=center[:1]), axis=1)
        path_distance = np.stack([np.zeros_like(center_delta), center_delta], axis=1)  # (n,2)
        return Track(path_distance=path_distance, path_nodes=path_nodes, path_width=width)

    # Nothing worked
    raise KeyError(
        f"Could not build Track for episode {ep}. "
        f"Expected one of: (path_distance,path_nodes,path_width) in info.npz; "
        f"track.npz with those keys; or (center,width)."
    )


class RoadDataset(Dataset):
    """
    SuperTux planning dataset.

    Supports:
      A) split-level metadata: <split>/info.npz
      B) per-episode metadata: <split>/<episode>/info.npz
    """

    def __init__(self, split_path: str, transform_pipeline: str = "default"):
        self.split_path = Path(split_path)

        # Discover episodes
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

        # Per-episode caches
        self.episode_frames: List[Dict[str, np.ndarray]] = []
        self.episode_track: List[Track] = []
        self.frame_index: List[Tuple[int, int]] = []  # (episode_idx, frame_idx)

        for ei, ep in enumerate(self.episode_dirs):
            info = np.load(ep / "info.npz", allow_pickle=True)

            # Track for this episode (robust to missing keys)
            track = _track_from_any(ep)
            self.episode_track.append(track)

            # Frames dict
            frames = _frames_from_info(info)
            self.episode_frames.append(frames)

            # Number of frames
            if "location" in frames:
                num_frames = len(frames["location"])
            elif "num_frames" in info.files:
                num_frames = int(info["num_frames"])
            elif "frames" in info.files and hasattr(info["frames"].item(), "__len__"):
                num_frames = len(info["frames"].item())
            else:
                raise ValueError(f"Cannot infer number of frames for episode '{ep}'")

            for fi in range(num_frames):
                self.frame_index.append((ei, fi))

        # Compose transform
        if transform_pipeline == "default":
            # Image + state
            self.transform = road_transforms.Compose([
                road_transforms.ImageLoader(None),  # episode path set per-sample
                road_transforms.EgoTrackProcessor(track=self.episode_track[0]),
                road_transforms.NormalizeImage(),
                road_transforms.ToTensor(),
            ])
        elif transform_pipeline == "state_only":
            # State only
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
            "_idx": fi,
            "_frames": frames,
            "episode_path": str(episode_path),
        }

        # Point the transform to the current episode and current track
        for t in getattr(self.transform, "transforms", []):
            if isinstance(t, road_transforms.ImageLoader):
                t.episode_path = Path(episode_path)
            if isinstance(t, (road_transforms.EgoTrackProcessor,)):
                t.track = track

        sample = self.transform(sample)
        return sample
