
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from . import road_transforms
from .road_utils import Track

class RoadDataset(Dataset):
    """SuperTux dataset for road detection. Supports split-level or per-episode info.npz."""
    def __init__(self, split_path: str, transform_pipeline: str = "default"):
        self.split_path = Path(split_path)
        root_info = self.split_path / "info.npz"
        if root_info.exists():
            self.episode_dirs = [self.split_path]
        else:
            self.episode_dirs = sorted(
                [d for d in self.split_path.iterdir() if d.is_dir() and (d / "info.npz").exists()]
            )
            if len(self.episode_dirs) == 0:
                raise FileNotFoundError(f"No info.npz under '{self.split_path}' or any episode subfolder.")
        self.track = Track()
        if transform_pipeline == "default":
            self.transform = road_transforms.Compose([
                road_transforms.ImageLoader(None),
                road_transforms.EgoTrackProcessor(self.track),
                road_transforms.NormalizeImage(),
                road_transforms.ToTensor(),
            ])
        elif transform_pipeline == "state_only":
            self.transform = road_transforms.Compose([
                road_transforms.EgoTrackProcessor(self.track),
                road_transforms.ToTensor(),
            ])
        else:
            raise ValueError(f"Unknown transform_pipeline: {transform_pipeline}")
        self.frame_index = []
        self.episode_infos = []
        for ei, ep in enumerate(self.episode_dirs):
            info = np.load(ep / "info.npz", allow_pickle=True)
            self.episode_infos.append(info)
            num_frames = int(info["num_frames"]) if "num_frames" in info.files else len(info["frames"])
            for fi in range(num_frames):
                self.frame_index.append((ei, fi))
    def __len__(self):
        return len(self.frame_index)
    def __getitem__(self, idx: int) -> dict:
        ei, fi = self.frame_index[idx]
        episode_path = self.episode_dirs[ei]
        sample = {"episode_path": str(episode_path), "frame_idx": fi}
        # Update ImageLoader episode path dynamically
        for t in getattr(self.transform, "transforms", []):
            if isinstance(t, road_transforms.ImageLoader):
                t.episode_path = Path(episode_path)
        sample = self.transform(sample)
        return sample
