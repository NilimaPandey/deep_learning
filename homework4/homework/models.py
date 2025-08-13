
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent

# -------------- Linear --------------
class LinearPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3):
        super().__init__()
        in_dim = n_track * 4
        out_dim = n_waypoints * 2
        self.fc = nn.Linear(in_dim, out_dim)
        self.n_waypoints = n_waypoints
    def forward(self, track_left, track_right):
        b = track_left.size(0)
        x = torch.cat([track_left, track_right], dim=1).reshape(b, -1)
        out = self.fc(x)
        return out.view(b, self.n_waypoints, 2)

# ---------------- MLP ----------------
class MLPPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3, hidden=(128, 256, 128)):
        super().__init__()
        in_dim = n_track * 4
        out_dim = n_waypoints * 2
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)
        self.n_waypoints = n_waypoints
    def forward(self, track_left, track_right):
        b = track_left.size(0)
        x = torch.cat([track_left, track_right], dim=1).reshape(b, -1)
        out = self.net(x)
        return out.view(b, self.n_waypoints, 2)

# -------- Transformer (decoder) ------
class TransformerPlanner(nn.Module):
    """
    State-only planner: encodes left/right track points, decodes N waypoints.
    """
    def __init__(self, n_track: int = 10, n_waypoints: int = 3,
                 d_model=256, nhead=8, num_layers=4, ff=512, dropout=0.1):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # project (x, y) -> d_model
        self.input_proj = nn.Linear(2, d_model)

        # positional (by index) and side (left/right) embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, n_track, d_model) * 0.02)
        self.side_embed = nn.Embedding(2, d_model)  # 0 = left, 1 = right

        # encode concatenated [left; right] track tokens
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # learned queries for each waypoint
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # cross-attend queries to encoded memory
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(d_model, 2)

    def forward(self, track_left, track_right):
        """
        track_left/right: (B, n_track, 2)
        returns: (B, n_waypoints, 2)
        """
        B, N, _ = track_left.shape

        # embeddings for left/right
        tl = self.input_proj(track_left) + self.pos_embed[:, :N, :] + self.side_embed.weight[0].view(1, 1, -1)
        tr = self.input_proj(track_right) + self.pos_embed[:, :N, :] + self.side_embed.weight[1].view(1, 1, -1)

        mem = torch.cat([tl, tr], dim=1)          # (B, 2N, d_model)
        mem = self.encoder(mem)                   # encode the track context

        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, n_wp, d_model)
        hs = self.decoder(q, mem)                 # (B, n_wp, d_model)
        return self.out_proj(hs)                  # (B, n_wp, 2)


# ---------------- CNN ----------------
# --- in homework/models.py ---
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.down is not None:
            x = self.down(x)
        return self.relu(x + y)

# --- homework/models.py ---
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.down is not None:
            x = self.down(x)
        return self.relu(x + y)

class CNNPlanner(nn.Module):
    """
    Image-only planner: image (B,3,H,W) -> waypoints (B, n_wp, 2)
    Assumes the dataloader already converted to CHW float and normalized (ImageNet mean/std).
    """
    def __init__(self, n_waypoints: int = 3):
        super().__init__()
        self.n_waypoints = n_waypoints

        # Stem
        layers = [
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),  # 1/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        # Residual stages at 1/4, 1/8, 1/16
        channels = [32, 64, 128, 256]
        strides  = [1,   2,   2,   2]
        in_ch = 32
        for out_ch, s in zip(channels, strides):
            layers += [BasicBlock(in_ch, out_ch, stride=s), BasicBlock(out_ch, out_ch, stride=1)]
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.backbone(image)                # already normalized by pipeline
        out = self.head(x)
        return out.view(image.size(0), self.n_waypoints, 2)

MODEL_FACTORY = {
    'linear_planner': LinearPlanner,
    'mlp_planner': MLPPlanner,
    'transformer_planner': TransformerPlanner,
    'cnn_planner': CNNPlanner,       # <-- ensure this exists
}


def build_model(name: str) -> nn.Module:
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model: {name}. Options: {list(MODEL_FACTORY)}")
    return MODEL_FACTORY[name]()

def save_model(model: nn.Module) -> str:
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

def load_model(
    model_name: str,
    with_weights: bool = True,
    map_location: str | torch.device | None = None,
) -> nn.Module:
    """
    Grader-compatible loader.
    - model_name: one of {'linear_planner','mlp_planner','transformer_planner','cnn_planner'}
    - with_weights: if True, tries to load homework/<model_name>.th if it exists
    - map_location: device for checkpoint loading; defaults to 'cpu' for grader
    Returns model in eval() mode on CPU (grader runs on CPU).
    """
    device = map_location or "cpu"
    model = build_model(model_name)

    if with_weights:
        ckpt_path = HOMEWORK_DIR / f"{model_name}.th"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            # be tolerant of minor key mismatches
            try:
                model.load_state_dict(state, strict=True)
            except Exception:
                model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model

