
from __future__ import annotations
import torch

class PlannerMetric:
    """Tracks longitudinal (x) and lateral (y) L1 errors over masked waypoints."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum_abs_x = 0.0
        self.sum_abs_y = 0.0
        self.count = 0
    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
        diff = (preds - targets).abs()
        mask2 = mask.unsqueeze(-1).to(diff.dtype)
        diff = diff * mask2
        self.sum_abs_x += float(diff[..., 0].sum().item())
        self.sum_abs_y += float(diff[..., 1].sum().item())
        self.count += int(mask.sum().item())
    def compute(self) -> dict:
        if self.count == 0:
            return {"l1_error": 0.0, "longitudinal_error": 0.0, "lateral_error": 0.0, "num_samples": 0}
        longitudinal = self.sum_abs_x / self.count
        lateral = self.sum_abs_y / self.count
        return {
            "l1_error": longitudinal + lateral,
            "longitudinal_error": longitudinal,
            "lateral_error": lateral,
            "num_samples": self.count,
        }
