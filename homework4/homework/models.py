
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
    def __init__(self, n_track: int = 10, n_waypoints: int = 3, d_model=128, nhead=4, num_layers=2, ff=256, dropout=0.1):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.input_proj = nn.Linear(2, d_model)
        self.pos_left = nn.Parameter(torch.randn(1, n_track, d_model) * 0.02)
        self.pos_right = nn.Parameter(torch.randn(1, n_track, d_model) * 0.02)
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=ff, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 2)
    def forward(self, track_left, track_right):
        tl = self.input_proj(track_left) + self.pos_left
        tr = self.input_proj(track_right) + self.pos_right
        mem = torch.cat([tl, tr], dim=1)
        q = self.query_embed.weight.unsqueeze(0).expand(track_left.size(0), -1, -1)
        hs = self.decoder(q, mem)
        return self.out_proj(hs)

# ---------------- CNN ----------------
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
    def __init__(self, n_waypoints: int = 3):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.input_mean = torch.tensor([0.485, 0.456, 0.406])
        self.input_std = torch.tensor([0.229, 0.224, 0.225])
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, s=2),
            ConvBlock(32, 32),
            ConvBlock(32, 64, s=2),
            ConvBlock(64, 64),
            ConvBlock(64, 128, s=2),
            ConvBlock(128, 128),
            ConvBlock(128, 256, s=2),
            ConvBlock(256, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_waypoints * 2),
        )
    def forward(self, image):
        x = (image - self.input_mean[None, :, None, None].to(image.device)) / self.input_std[None, :, None, None].to(image.device)
        feats = self.backbone(x)
        out = self.head(feats)
        return out.view(image.size(0), self.n_waypoints, 2)

MODEL_FACTORY = {
    'linear_planner': LinearPlanner,
    'mlp_planner': MLPPlanner,
    'transformer_planner': TransformerPlanner,
    'cnn_planner': CNNPlanner,
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

def load_model(model_name: str, checkpoint_path: str | None = None, map_location: str | None = None) -> nn.Module:
    device = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
