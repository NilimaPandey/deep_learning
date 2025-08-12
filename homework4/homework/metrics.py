
import numpy as np
import torch


class PlannerMetric:
    """Computes longitudinal and lateral errors for a planner.

    We assume labels and predictions are waypoints in the ego (vehicle) frame
    with shape (B, n_waypoints, 2) where the 2 dims are (forward-x, lateral-y).
    A boolean mask of shape (B, n_waypoints) indicates which targets are valid.
    """

    def __init__(self):
        self.l1_errors = []
        self.total = 0

    def reset(self):
        self.l1_errors = []
        self.total = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, labels: torch.Tensor, labels_mask: torch.Tensor):
        """Accumulate error for a batch.

        Args:
            preds:  (B, n_waypoints, 2) float tensor
            labels: (B, n_waypoints, 2) float tensor
            labels_mask: (B, n_waypoints) bool or 0/1 tensor
        """
        # Ensure shapes
        assert preds.ndim == 3 and preds.size(-1) == 2, f"preds shape {preds.shape} invalid"
        assert labels.shape == preds.shape, f"labels shape {labels.shape} must match preds"
        assert labels_mask.shape[:2] == preds.shape[:2], "mask must be (B, n_waypoints)"

        # Convert mask to float and expand to (B, n_waypoints, 2)
        mask = labels_mask.to(preds.dtype).unsqueeze(-1).expand_as(preds)

        # L1 absolute error on (longitudinal=x, lateral=y)
        abs_err = (preds - labels).abs() * mask  # (B, n_waypoints, 2)

        # Sum errors across batch & waypoints separately for the two axes
        # Result shape (2,)
        err_sum = abs_err.sum(dim=(0, 1))  # (2,)

        # Keep on CPU as numpy-friendly
        error_sum = err_sum.detach().cpu().numpy()
        self.l1_errors.append(error_sum)
        self.total += labels_mask.sum().item()

    def compute(self) -> dict[str, float]:
        error = np.stack(self.l1_errors, axis=0) if self.l1_errors else np.zeros((1, 2), dtype=float)
        longitudinal_error = error[:, 0].sum() / max(self.total, 1)
        lateral_error = error[:, 1].sum() / max(self.total, 1)
        l1_error = longitudinal_error + lateral_error

        return {
            "l1_error": float(l1_error),
            "longitudinal_error": float(longitudinal_error),
            "lateral_error": float(lateral_error),
            "num_samples": int(self.total),
        }
