"""
Re‑build the same architecture and load your *.pt* weights.
"""
from pathlib import Path
import torch
import torch.nn as nn

CHECKPOINT = Path(__file__).parent / "saved_models" / "history_mistake_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleHistoryBasedModel(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=3):
        super().__init__()
        self.history_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.response_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, h, r):
        h = self.history_proj(h)
        r = self.response_proj(r)
        return self.ff(torch.cat([h, r], dim=-1))


MODEL = SimpleHistoryBasedModel().to(DEVICE)
MODEL.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
MODEL.eval()
