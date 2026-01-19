import torch
from torch import Tensor


# ============================================================
# 1️⃣ EqNIO O(2) 特征构造（向量 / 标量拆分）
# ============================================================

def build_o2_features_from_imu(a: Tensor, w1: Tensor, w2: Tensor):
    """
    基于 EqNIO (O(2)) 的 IMU 特征构造方式。

    输入：
      a, w1, w2: (B, T, 3)
        - 已经完成重力对齐（gravity-aligned）
        - a: 加速度
        - w1, w2: 由角速度 ω 分解得到的两个普通向量（2-vector trick）

    输出：
      vec: (B, T, 2, 3)
        - O(2) 等变的二维向量特征
        - 包含 3 个 vector channel：
            [a_xy, w1_xy, w2_xy]

      scalars: (B, T, 9)
        - O(2) 不变的标量特征（C0^s = 9）：
            1) z 分量（3 个）
            2) xy 平面模长（3 个）
            3) 两两点积（3 个）

    说明：
      - 这是 EqNIO 论文和开源实现中 O(2) FrameNet 的标准输入形式
      - vec 用于等变网络
      - scalars 用于 gated nonlinearity 中的标量分支
    """
    # 取 xy 分量
    a_xy, w1_xy, w2_xy = a[..., :2], w1[..., :2], w2[..., :2]

    # 构造 O(2) 向量特征 (B,T,2,3)
    vec = torch.stack([a_xy, w1_xy, w2_xy], dim=-1)

    # z 分量（标量不变）
    z = torch.stack([a[..., 2], w1[..., 2], w2[..., 2]], dim=-1)

    # xy 平面模长（标量不变）
    n = torch.stack(
        [a_xy.norm(dim=-1), w1_xy.norm(dim=-1), w2_xy.norm(dim=-1)],
        dim=-1
    )

    # 两两点积（标量不变）
    d12 = (a_xy * w1_xy).sum(dim=-1)
    d13 = (a_xy * w2_xy).sum(dim=-1)
    d23 = (w1_xy * w2_xy).sum(dim=-1)
    dots = torch.stack([d12, d13, d23], dim=-1)

    # 拼接得到 C0^s = 9
    scalars = torch.cat([z, n, dots], dim=-1)
    return vec, scalars


# ============================================================
# 2️⃣ 使用 Frame 在 XY 平面进行 canonicalization
# ============================================================

def apply_frame_xy(x: Tensor, Fm: Tensor, inverse: bool = True) -> Tensor:
    """
    在 XY 平面上应用 2×2 Frame（canonical yaw frame）。

    输入：
      x:  (B, T, 3)
        - 任意 3D 序列（mag / acc / v1 / v2）

      Fm: (B, 2, 2)
        - 由 FrameNet 预测得到的正交矩阵

      inverse:
        - True  : 使用 F^{-1}（canonicalize）
        - False : 使用 F（反向变换）

    输出：
      x': (B, T, 3)
        - XY 分量被 Frame 旋转
        - Z 分量保持不变（重力对齐假设）
    """
    xy = x[..., :2]  # (B,T,2)

    # 对正交矩阵，逆等于转置
    M = Fm.transpose(-1, -2) if inverse else Fm

    xy2 = torch.einsum("bij,btj->bti", M, xy)
    return torch.cat([xy2, x[..., 2:3]], dim=-1)


# 语义别名（便于在 loss 中阅读）
def canonicalize_by_frame_xy(x: Tensor, Fm: Tensor, inverse: bool = True) -> Tensor:
    return apply_frame_xy(x, Fm, inverse=inverse)


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
