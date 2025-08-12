
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


# -----------------------------
# Part 1a: MLP Planner
# -----------------------------
class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = 2 * n_track * 2  # left (n,2) + right (n,2)
        output_dim = n_waypoints * 2

        layers = []
        dims = [input_dim] + [hidden] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        # track_*: (B, n_track, 2)
        B = track_left.size(0)
        x = torch.cat([track_left, track_right], dim=1)  # (B, 2*n_track, 2)
        x = x.reshape(B, -1)  # (B, 2*n_track*2)
        out = self.net(x)  # (B, n_waypoints*2)
        return out.view(B, self.n_waypoints, 2)


# -----------------------------
# Part 1b: Transformer Planner
# -----------------------------
class PosEncMLP(nn.Module):
    """Small MLP to lift 2D coordinates to a higher dim feature space."""
    def __init__(self, in_dim=2, emb_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.proj(x)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.input_proj = PosEncMLP(2, d_model)  # encode (x,y) -> d_model

        # Learned waypoint queries
        self.queries = nn.Embedding(n_waypoints, d_model)

        # We'll treat inputs as the memory (keys/values), queries as target tokens.
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # Predict (x,y) for each waypoint
        self.out = nn.Linear(d_model, 2)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        # Concatenate lane points
        # Inputs: (B, n_track, 2)
        x = torch.cat([track_left, track_right], dim=1)  # (B, 2*n_track, 2)

        # Encode to d_model features
        mem = self.input_proj(x)  # (B, 2*n_track, d_model)

        # Prepare learned queries
        B = x.size(0)
        q_idx = torch.arange(self.n_waypoints, device=x.device)  # (n_waypoints,)
        tgt = self.queries(q_idx).unsqueeze(0).expand(B, -1, -1)  # (B, n_waypoints, d_model)

        # Cross-attend: queries (tgt) attend to memory (mem)
        dec = self.decoder(tgt=tgt, memory=mem)  # (B, n_waypoints, d_model)
        preds = self.out(dec)  # (B, n_waypoints, 2)
        return preds


# -----------------------------
# Part 2: CNN Planner
# -----------------------------
class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints: int = 3):
        super().__init__()
        self.n_waypoints = n_waypoints

        # A lightweight CNN backbone (works on 3x96x128 images)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 3x96x128 -> 32x48x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x24x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x12x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256x6x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 256x1x1
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, n_waypoints * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 96, 128) normalized image
        feats = self.backbone(x)
        out = self.head(feats)  # (B, n_waypoints*2)
        B = x.size(0)
        return out.view(B, self.n_waypoints, 2)


# -----------------------------
# Utilities
# -----------------------------
def save_model(model: torch.nn.Module) -> Path:
    model_name: Optional[str] = None
    if isinstance(model, MLPPlanner):
        model_name = "mlp_planner"
    elif isinstance(model, TransformerPlanner):
        model_name = "transformer_planner"
    elif isinstance(model, CNNPlanner):
        model_name = "cnn_planner"

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """Naive way to estimate model size (float32 params assumed)."""
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
from typing import Union
def load_model(what: Union[str, Path], device: torch.device | str | None = None) -> nn.Module:
    """
    Load a trained planner.

    Args:
        what: Either a model name {"mlp_planner","transformer_planner","cnn_planner"}
              or a path to a checkpoint file (*.th).
        device: torch.device or string ("cuda"/"cpu"), optional.

    Returns:
        An nn.Module moved to `device` and set to eval().
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_from_name(name: str) -> nn.Module:
        name = name.lower()
        if name == "mlp_planner":
            return MLPPlanner()
        if name == "transformer_planner":
            return TransformerPlanner()
        if name == "cnn_planner":
            return CNNPlanner()
        raise ValueError(f"Unknown model name: {name}")

    p = Path(what)
    if p.exists():  # treat as checkpoint path
        # Heuristic: infer class from filename; fallback to MLP
        fname = p.name.lower()
        if "transformer" in fname:
            model = TransformerPlanner()
        elif "cnn" in fname:
            model = CNNPlanner()
        elif "mlp" in fname or "planner" in fname:
            model = MLPPlanner()
        else:
            # If your grader passes an explicit model name separately, you can change this.
            model = MLPPlanner()
        state = torch.load(p, map_location=device)
        model.load_state_dict(state, strict=True)
    else:
        # treat `what` as a model name and load default checkpoint path
        model = _make_from_name(what)
        ckpt = HOMEWORK_DIR / f"{what.lower()}.th"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=True)

    model.to(device).eval()
    return model
