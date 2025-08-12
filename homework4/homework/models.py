
from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Directory to save checkpoints
HOMEWORK_DIR = Path(__file__).resolve().parent

# -------------------------
# MLP Planner
# -------------------------


class LinearPlanner(nn.Module):
    """
    Linear baseline: flattens left/right tracks and applies a single Linear layer.
    """
    def __init__(self, n_track: int = 10, n_waypoints: int = 3):
        super().__init__()
        in_dim = n_track * 2 * 2
        out_dim = n_waypoints * 2
        self.fc = nn.Linear(in_dim, out_dim)
        self.n_waypoints = n_waypoints

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        b = track_left.size(0)
        x = torch.cat([track_left, track_right], dim=1).reshape(b, -1)
        out = self.fc(x)  # (B, 2*n_waypoints)
        return out.view(b, self.n_waypoints, 2)


class MLPPlanner(nn.Module):
    """
    Predict n_waypoints 2D future positions from left/right lane boundaries.
    Inputs:
        track_left:  (B, n_track, 2)
        track_right: (B, n_track, 2)
    Output:
        waypoints: (B, n_waypoints, 2)
    """
    def __init__(self, n_track: int = 10, n_waypoints: int = 3, hidden: Tuple[int, ...] = (128, 256, 128)):
        super().__init__()
        in_dim = n_track * 2 * 2  # left/right * (x,y)
        out_dim = n_waypoints * 2

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)
        self.n_waypoints = n_waypoints

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        b = track_left.size(0)
        x = torch.cat([track_left, track_right], dim=1)  # (B, 2*n_track, 2)
        x = x.reshape(b, -1)  # (B, 4*n_track)
        out = self.net(x)  # (B, 2*n_waypoints)
        return out.view(b, self.n_waypoints, 2)


# -------------------------
# Transformer Planner
# -------------------------

class TransformerPlanner(nn.Module):
    """
    Perceiver-style cross-attention: learned waypoint queries attend over lane boundary tokens.
    """
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints

        # Encode inputs (left/right tracks) -> tokens of dim d_model
        self.input_proj = nn.Linear(2, d_model)  # (x,y) -> token
        self.pos_left = nn.Parameter(torch.randn(1, n_track, d_model) * 0.02)
        self.pos_right = nn.Parameter(torch.randn(1, n_track, d_model) * 0.02)

        # Learned queries for n_waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(d_model, 2)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        # tracks: (B, n_track, 2)
        tl = self.input_proj(track_left) + self.pos_left  # (B, n_track, d)
        tr = self.input_proj(track_right) + self.pos_right  # (B, n_track, d)
        mem = torch.cat([tl, tr], dim=1)  # (B, 2*n_track, d)

        b = track_left.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (B, n_waypoints, d)

        hs = self.decoder(tgt=queries, memory=mem)  # (B, n_waypoints, d)
        out = self.out_proj(hs)  # (B, n_waypoints, 2)
        return out


# -------------------------
# CNN Planner
# -------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CNNPlanner(nn.Module):
    """
    Predict n_waypoints 2D positions from image (B, 3, 96, 128).
    """
    def __init__(self, n_waypoints: int = 3):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.input_mean = torch.tensor([0.485, 0.456, 0.406])
        self.input_std = torch.tensor([0.229, 0.224, 0.225])

        self.backbone = nn.Sequential(
            ConvBlock(3, 32, s=2),    # 48x64
            ConvBlock(32, 32),
            ConvBlock(32, 64, s=2),   # 24x32
            ConvBlock(64, 64),
            ConvBlock(64, 128, s=2),  # 12x16
            ConvBlock(128, 128),
            ConvBlock(128, 256, s=2), # 6x8
            ConvBlock(256, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Normalize to ImageNet stats (image expected in [0,1])
        x = (image - self.input_mean[None, :, None, None].to(image.device)) / self.input_std[None, :, None, None].to(image.device)
        feats = self.backbone(x)
        out = self.head(feats)  # (B, 2*n_waypoints)
        return out.view(image.size(0), self.n_waypoints, 2)


# -------------------------
# Factory and Save
# -------------------------

MODEL_FACTORY = {
    'linear_planner': LinearPlanner,
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def build_model(name: str) -> nn.Module:
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model: {name}. Options: {list(MODEL_FACTORY)}")
    return MODEL_FACTORY[name]()


def save_model(model: nn.Module) -> str:
    """
    Save model state dict to homework/<model_name>.th and return path.
    """
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n
            break
    if model_name is None:
        raise ValueError(f"Unsupported model type: {type(model)}")
    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return str(output_path)
