# network/losses.py
import torch
import torch.nn as nn

class WeightedSmoothL1(nn.Module):
    def __init__(self, beta=0.05, w_x=1.0, w_y=1.3):
        super().__init__()
        self.beta = float(beta)
        self.register_buffer("w", torch.tensor([w_x, w_y], dtype=torch.float32))

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = diff.abs()
        beta = self.beta
        loss = torch.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
        return (loss * self.w.view(1, 2)).mean()
    

class L2SmoothLoss(nn.Module):
    def __init__(self, beta=0.5):  # beta单位取决于你y的归一化尺度；如果是[0,1]可先0.02~0.1
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        diff = pred - target
        l2 = torch.sqrt((diff * diff).sum(dim=1) + 1e-12)  # (B,)
        beta = self.beta
        loss = torch.where(l2 < beta, 0.5 * l2 * l2 / beta, l2 - 0.5 * beta)
        return loss.mean()

