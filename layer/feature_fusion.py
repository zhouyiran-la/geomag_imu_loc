import torch
import torch.nn.functional as F
from torch import nn, Tensor


class ConcatFusionHead(nn.Module):
    """
    最稳的融合方式：直接 concat 后 MLP 回归
    """
    def __init__(self, mag_dim: int, imu_dim: int, hidden: int = 256, out_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(mag_dim + imu_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, mag_feat: Tensor, imu_feat: Tensor) -> Tensor:
        x = torch.cat([mag_feat, imu_feat], dim=-1)
        return self.net(x)