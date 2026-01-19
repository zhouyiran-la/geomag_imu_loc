import torch
import torch.nn.functional as F
from torch import nn, Tensor

# ============================================================
# 2) IMU Encoder: RONIN-like ResNet1D (单尺度输出一个向量)
# ============================================================

class ResBlock1D(nn.Module):
    def __init__(self, c: int, kernel: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(c, c, kernel_size=kernel, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(c, c, kernel_size=kernel, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(c)
        self.norm2 = nn.BatchNorm1d(c)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,C,T)
        res = x
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.relu(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.dropout(y)

        y = F.relu(res + y)
        return y


class RoninLikeImuEncoder(nn.Module):
    """
    输入: (B,T,9) 其中 9 = [acc(3), v1(3), v2(3)]（已 canonicalize）
    输出: imu_feat (B, D_out)

    说明：
      - 不做 dataset mean/std normalize（符合 EqNIO 物理假设）
      - 用一个轻量 ResNet1D 做时序建模 + 全局池化
    """
    def __init__(
        self,
        in_dim: int = 9,
        c: int = 128,
        blocks: int = 6,
        kernel: int = 5,
        dropout: float = 0.1,
        out_dim: int = 64,  # 建议与 mag feature_dim 对齐或更小都行
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, c, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(c),
            nn.ReLU(),
        )
        self.res = nn.ModuleList([
            ResBlock1D(c, kernel=kernel, dilation=2 ** (i % 4), dropout=dropout) for i in range(blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(c, c),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(c, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,T,9) -> (B,9,T)
        x = x.transpose(1, 2)
        x = self.stem(x)
        for blk in self.res:
            x = blk(x)

        # 全局平均池化 -> (B,C)
        x = x.mean(dim=-1)
        x = self.head(x)
        return x