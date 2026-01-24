import torch
from typing import TypeVar
from network.utils import canonicalize_by_frame_xy
from torch import Tensor

T = TypeVar("T")
def move_to_device(x: T, device: torch.device) -> T:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)  # type: ignore[return-value]
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}  # type: ignore[return-value]
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)  # type: ignore[return-value]
    return x


# ============================================================
# 3️⃣ Canonical Consistency Loss（EqNIO 核心）
# ============================================================

def canonical_consistency_loss(
    *,
    Fm: Tensor,
    Fm_aug: Tensor,
    mag: Tensor,
    acc: Tensor,
    v1: Tensor,
    v2: Tensor,
    mag_aug: Tensor,
    acc_aug: Tensor,
    v1_aug: Tensor,
    v2_aug: Tensor,
    w_mag: float = 1.0,
    w_imu: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """
    EqNIO 的 canonical consistency loss（无需 heading GT）。

    核心思想：
        对同一条轨迹的原始输入和O(2) 增强输入，
        在各自预测的 frame 下 canonicalize 后，结果应一致。

    数学形式：
        L = || F^{-1} x  -  F_aug^{-1} x_aug ||_1

    输入：
      Fm / Fm_aug:
        - (B,2,2)
        - 原始 / 增强 forward 得到的 frame

      mag, acc, v1, v2:
        - (B,T,3)
        - 原始输入

      *_aug:
        - 对应的 O(2) 增强输入

      w_mag / w_imu:
        - 地磁与 IMU consistency loss 的权重

    输出：
      loss:
        - 标量（mean / sum）
        - 或 (B,)（reduction="none"）
    """
    # 原始输入 canonicalization
    mag_c = canonicalize_by_frame_xy(mag, Fm, inverse=True)
    acc_c = canonicalize_by_frame_xy(acc, Fm, inverse=True)
    v1_c  = canonicalize_by_frame_xy(v1,  Fm, inverse=True)
    v2_c  = canonicalize_by_frame_xy(v2,  Fm, inverse=True)

    # 增强输入 canonicalization
    mag_c_aug = canonicalize_by_frame_xy(mag_aug, Fm_aug, inverse=True)
    acc_c_aug = canonicalize_by_frame_xy(acc_aug, Fm_aug, inverse=True)
    v1_c_aug  = canonicalize_by_frame_xy(v1_aug,  Fm_aug, inverse=True)
    v2_c_aug  = canonicalize_by_frame_xy(v2_aug,  Fm_aug, inverse=True)

    # L1 距离（逐样本）
    loss_mag = (mag_c - mag_c_aug).abs().mean(dim=(1, 2))
    loss_imu = (
        (acc_c - acc_c_aug).abs().mean(dim=(1, 2))
        + (v1_c - v1_c_aug).abs().mean(dim=(1, 2))
        + (v2_c - v2_c_aug).abs().mean(dim=(1, 2))
    ) / 3.0

    loss = w_mag * loss_mag + w_imu * loss_imu

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"未知 reduction 类型: {reduction}")

def forward_with_yaw_pair(model, batch: dict):
    """
    训练辅助函数：对同一 batch 进行两次 forward。

    约定：
      batch 必须包含：
        - x_mag, x_acc, x_v1, x_v2
        - batch["aug"] 中包含增强版本：
            aug["x_mag"], aug["x_acc"], aug["x_v1"], aug["x_v2"]

    返回：
      pred, extras:
        - 原始 forward 的输出与中间信息（extras["Fm"]）

      pred_aug, extras_aug:
        - 增强 forward 的输出与中间信息
    """
    pred, extras = model(
        batch["x_mag"],
        batch["x_acc"],
        batch["x_v1"],
        batch["x_v2"]
    )

    aug = batch["aug"]
    pred_aug, extras_aug = model(
        aug["x_mag"],
        aug["x_acc"],
        aug["x_v1"],
        aug["x_v2"]
    )

    return pred, extras, pred_aug, extras_aug