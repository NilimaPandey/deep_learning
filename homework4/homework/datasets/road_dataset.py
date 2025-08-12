from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from torch.utils.data import Dataset

from . import road_transforms
from .road_utils import Track


def _to3(x: np.ndarray) -> np.ndarray:
    """Ensure points are (N,3). If (N,2), append zeros z=0."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[-1] not in (2, 3):
        raise ValueError(f"Expected (N,2|3), got {x.shape}")
    if x.shape[-1] == 2:
        z = np.zeros((x.shape[0], 1), dtype=np.float32)
        x = np.concatenate([x, z], axis=1)
    return x


def _synthesize_track_from_center_width(center: np.ndarray, width: np.ndarray) -> Track:
    center = _to3(center)  # (N,3)
    width = np.asarray(width, dtype=np.float32).reshape(-1, 1)  # (N,1)
    # Minimal path_nodes: use center for both L/R nodes (Track only needs geometry along the path)
    path_nodes = np.stack([center, center], axis=1)  # (N,2,3)
    # Approx path_distance: previous-to-current segment length
    seg = np.linalg.norm(np.diff(center, axis=0, prepend=center[:1]), axis=1)
    path_distance = np.stack([np.zeros_like(seg), seg], axis=1).astype(np.float32)  # (N,2)
    return Track(path_distance=path_distance, path_nodes=path_nodes, path_width=width)


def _synthesize_track_from_left_right(left: np.ndarray, right: np.ndarray) -> Track:
    L = _to3(left)
    R = _to3(right)
    if L.shape != R.shape:
        n = min(L.shape[0], R.shape[0])
        L, R = L[:n], R[:n]
    center = 0.5 * (L + R)
    width = np.linalg.norm(R - L, axis=1, keepdims=True).astype(np.float32)
    path_nodes = np.stack([L, R], axis=1).astype(np.float32)  # (N,2,3)
    seg = np.linalg.norm(np.diff(center, axis=0, prepend=center[:1]), axis=1)
    path_distance = np.stack([np.zeros_like(seg), seg], axis=1).astype(np.float32)
    return Track(path_distance=path_distance, path_nodes=path_nodes, path_width=width)


def _maybe(info_like, *names) -> Optional[np.ndarray]:
    """Try multiple key names in an np.load handle or dict (case-insensitive)."""
    if hasattr(info_like, "files"):
        keys = {k.lower(): k for k in info_like.files}
        getter = lambda k: info_like[keys[k]]
    elif isinstance(info_like, dict):
        keys = {k.lower(): k for k in info_like.keys()}
        getter = lambda k: info_like[keys[k]]
    else:
        return None
    for n in names:
        n = n.lower()
        if n in keys:
            return getter(n)
    return None


def _track_from_any(ep: Path) -> Track:
    """Build a Track from various schema variants inside an episode folder."""
    info = np.load(ep / "info.npz", allow_pickle=True)

    # 1) Direct keys in info.npz
    pd = _maybe(info, "path_distance")
    pn = _maybe(info, "path_nodes")
    pw = _maybe(info, "path_width")
    if pd is not None and pn is not None and pw is not None:
        return Track(path_distance=pd, path_nodes=pn, path_width=pw)

    # 2) Nested 'track' dict inside info.npz
    track_obj = _maybe(info, "track")
    if track_obj is not None:
        try:
            track_dict = track_obj.item() if hasattr(track_obj, "item") else track_obj
            pd = _maybe(track_dict, "path_distance")
            pn = _maybe(track_dict, "path_nodes")
            pw = _maybe(track_dict, "path_width")
            if pd is not None and pn is not None and pw is not None:
                return Track(path_distance=pd, path_nodes=pn, path_width=pw)
            # center + width in track dict
            center = _maybe(track_dict, "center", "centerline", "path_center")
            width = _maybe(track_dict, "width", "path_width")
            if center is not None and width is not None:
                return _synthesize_track_from_center_width(center, width)
            # left/right in track dict
            left = _maybe(track_dict, "left", "left_boundary", "path_left", "left_nodes")
            right = _maybe(track_dict, "right", "right_boundary", "path_right", "right_nodes")
            if left is not None and right is not None:
                return _synthesize_track_from_left_right(left, right)
        except Exception:
            pass

    # 3) Separate track.npz
    track_npz = ep / "track.npz"
    if track_npz.exists():
        t = np.load(track_npz, allow_pickle=True)
        pd = _maybe(t, "path_distance")
        pn = _maybe(t, "path_nodes")
        pw = _maybe(t, "path_width")
        if pd is not None and pn is not None and pw is not None:
            return Track(path_distance=pd, path_nodes=pn, path_width=pw)
        center = _maybe(t, "center", "centerline", "path_center")
        width = _maybe(t, "width", "path_width")
        if center is not None and width is not None:
            return _synthesize_track_from_center_width(center, width)
        left = _maybe(t, "left", "left_boundary", "path_left", "left_nodes")
        right = _maybe(t, "right", "right_boundary", "path_right", "right_nodes")
        if left is not None and right is not None:
            return _synthesize_track_from_left_right(left, right)

    # 4) Fallback: center/width directly in info.npz
    center = _maybe(info, "center", "centerline", "path_center")
    width = _maybe(info, "width", "path_width")
    if center is not None and width is not None:
        return _synthesize_track_from_center_width(center, width)

    # 5) Fallback: left/right directly in info.npz
    left = _maybe(info, "left", "left_boundary", "path_left", "left_nodes")
    right = _maybe(info, "right", "right_boundary", "path_right", "right_nodes")
    if left is not None and right is not None:
        return _synthesize_track_from_left_right(left, right)

    raise KeyError(
        f"Could not build Track for episode {ep}. "
        f"Tried: (path_distance,path_nodes,path_width), nested track dict, track.npz, "
        f"(center,width), (left,right) under several common names."
    )


def _frames_from_info(info, require_images: bool) -> Dict[str, np.ndarray]:
    """
    Extract only what the chosen pipeline needs.
    - For 'default' (image+state): need 'front','location','distance_down_track'
    - For 'state_only': only 'location' is minimally required; 'distance_down_track' can be derived.
    """
    frames = {}
    # Prefer top-level keys
    if require_images:
        for k in ("front", "location", "distance_down_track"):
            if k in info.files:
                frames[k] = info[k]
    else:
        for k in ("location", "distance_down_track"):
            if k in info.files:
                frames[k] = info[k]

    # Check nested frames dict
    if "frames" in info.files:
        try:
            fr = info["frames"].item()
            if isinstance(fr, dict):
                needed = ("front", "location", "distance_down_track") if require_images else ("location", "distance_down_track")
                for k in needed:
                    if k not in frames and k in fr:
                        frames[k] = fr[k]
        except Exception:
            pass

    # Minimal checks
    needed = ("front", "location") if require_images else ("location",)
    missing = [k for k in needed if k not in frames]
    if missing:
        raise KeyError(f"Missing required frame keys {missing} in info.npz")

    return frames


class RoadDataset(Dataset):
    """
    SuperTux planning dataset (split-level or per-episode).
    """

    def __init__(self, split_path: str, transform_pipeline: str = "default"):
        self.split_path = Path(split_path)
        self.require_images = (transform_pipeline == "default")

        # Find episodes
        root_info = self.split_path / "info.npz"
        if root_info.exists():
            self.episode_dirs = [self.split_path]
        else:
            self.episode_dirs = sorted(d for d in self.split_path.iterdir() if d.is_dir() and (d / "info.npz").exists())
            if not self.episode_dirs:
                raise FileNotFoundError(f"No info.npz under '{self.split_path}' or any episode subfolder.")

        # Build per-episode caches
        self.episode_frames: List[Dict[str, np.ndarray]] = []
        self.episode_track: List[Track] = []
        self.frame_index: List[Tuple[int, int]] = []

        for ei, ep in enumerate(self.episode_dirs):
            info = np.load(ep / "info.npz", allow_pickle=True)

            # Track
            track = _track_from_any(ep)
            self.episode_track.append(track)

            # Frames needed for pipeline
            frames = _frames_from_info(info, require_images=self.require_images)
            self.episode_frames.append(frames)

            # Frame count
            if "location" in frames:
                num_frames = len(frames["location"])
            elif "num_frames" in info.files:
                num_frames = int(info["num_frames"])
            else:
                raise ValueError(f"Cannot infer number of frames for episode '{ep}'")

            for fi in range(num_frames):
                self.frame_index.append((ei, fi))

        # Build transform pipeline
        if transform_pipeline == "default":
            self.transform = road_transforms.Compose([
                road_transforms.ImageLoader(None),  # episode set per-sample
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

        sample = {"_idx": fi, "_frames": frames, "episode_path": str(episode_path)}

        # Update per-sample episode & track inside transforms
        for t in getattr(self.transform, "transforms", []):
            if isinstance(t, road_transforms.ImageLoader):
                t.episode_path = Path(episode_path)
            if isinstance(t, (road_transforms.EgoTrackProcessor,)):
                t.track = track

        return self.transform(sample)
