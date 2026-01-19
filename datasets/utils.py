import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets.transforms import DefaultTransform, MagneticGradientTransform, ComposeTransform


def load_all_npz_files(data_dir, pattern=".npz", use_imu=True):
    """
    批量加载指定目录下的 .npz 文件，并拼接为统一数组。
    - 支持传入 '.npz'（自动补全为 '*.npz'）或完整通配（如 '*.npz' / '**/*.npz'）
    - 若未找到文件，抛出明确异常以便排查路径/模式问题。
    """
    X_MAG_list, X_IMU_list, y_list = [], [], []
    path = Path(data_dir)

    # 规范化 pattern（兼容传入 '.npz' 场景）
    patt = pattern
    if patt.startswith('.') and not patt.startswith('*.'):
        patt = f"*{patt}"

    # 查找文件并排序（保证加载顺序稳定）
    files = sorted(path.glob(patt))

    # 如果使用递归匹配（包含 "**/"），改用 rglob
    if not files and ('**/' in patt or patt.startswith('**')):
        files = sorted(path.rglob(patt.replace('**/', '')))

    if not files:
        raise FileNotFoundError(
            f"未在目录 {path} 下按模式 '{pattern}' 匹配到任何文件。\n"
            f"请检查: 1) data_dir 是否正确 2) pattern 是否应为 '*.npz' 或 '**/*.npz' 3) 运行时工作目录。"
        )

    for file in files:
        print(f"正在加载 {str(file)} 文件")
        data = np.load(file)
        X_mag = data["X_mag"]
        print(f"{str(file.name)} - X_mag.shape = {X_mag.shape}")
        X_MAG_list.append(X_mag)

        y = data["y"]
        print(f"{str(file.name)} - y.shape = {y.shape}")
        y_list.append(y)

        if use_imu and "X_imu" in data:
            X_IMU_list.append(data["X_imu"])

    if len(X_MAG_list) == 0 or len(y_list) == 0:
        raise RuntimeError("匹配到文件但未成功读取到 X_mag 或 y 数据，请检查 .npz 内部键名是否为 'X_mag' 与 'y'")

    X_MAG = np.concatenate(X_MAG_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if use_imu and len(X_IMU_list) > 0:
        X_IMU = np.concatenate(X_IMU_list, axis=0)
    else:
        X_IMU = None

    return X_MAG, X_IMU, y


# def build_transform(input_key):
#     if input_key == "x_mag_grad":
#         return ComposeTransform([
#             MagneticGradientTransform(),
#             DefaultTransform(),
#         ])
#     else:
#         return DefaultTransform()
    

def compute_train_stats_from_csv_files(file_paths, mag_cols, pos_cols, eps=1e-6):
    """
    只用 train 文件计算统计量：x_mean/x_std/y_mean/y_std/y_min/y_max
    """
    n = 0
    sum_x = None
    sum_x2 = None
    y_list = []

    usecols = list(mag_cols) + list(pos_cols)

    for p in file_paths:
        df = pd.read_csv(p, usecols=usecols)
        x = df[mag_cols].to_numpy(dtype=np.float32)  # (T,C)
        y = df[pos_cols].to_numpy(dtype=np.float32)  # (T,2)

        if sum_x is None:
            sum_x = x.sum(axis=0)
            sum_x2 = (x * x).sum(axis=0)
        else:
            sum_x += x.sum(axis=0)
            sum_x2 += (x * x).sum(axis=0)

        n += x.shape[0]
        y_list.append(y)

    if n <= 0:
        raise RuntimeError("train files empty: cannot compute stats")

    x_mean = (sum_x / n).astype(np.float32) # type: ignore
    x_var = (sum_x2 / n - x_mean * x_mean).astype(np.float32) # type: ignore
    x_std = np.sqrt(np.maximum(x_var, 0.0)).astype(np.float32)
    x_std = np.maximum(x_std, eps)

    Y = np.concatenate(y_list, axis=0)
    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": Y.mean(axis=0).astype(np.float32),
        "y_std": np.maximum(Y.std(axis=0).astype(np.float32), eps),
        "y_min": Y.min(axis=0).astype(np.float32),
        "y_max": Y.max(axis=0).astype(np.float32),
    }
    return stats


def norm_y(y_norm_mode:str, y_true, pf=None, stats=None):
    """
    Docstring for norm_y
    Args:
        y_norm_mode: str 标准化模式 global_zscore global_minmax per_file_minmax
        y_true: nd.array (2,)
        pf: dict {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max} 
        stats: global*模式下 训练文件的统计量
    Return：
        y_train
        y_stats
    """
    if y_norm_mode == "none":
        y_train = y_true
        y_stats = None
        return y_train, y_stats
    elif y_norm_mode == "global_zscore":
        assert stats, "stats不能为None"
        y_train = (y_true - stats["y_mean"]) / (stats["y_std"] + 1e-6)
        y_stats = np.array([*stats["y_mean"], *stats["y_std"]], dtype=np.float32)  # optional
        return y_train, y_stats
    elif y_norm_mode == "global_minmax":
        assert stats, "stats不能为None"
        y_train = (y_true - stats["y_min"]) / (stats["y_max"] - stats["y_min"] + 1e-6)
        y_stats = np.array([*stats["y_min"], *stats["y_max"]], dtype=np.float32)
        return y_train, y_stats
    elif y_norm_mode == "per_file_minmax":
        assert pf, "每个文件统计量pf不能为None"
        # 用 per-file stats 做 minmax
        y_denx = (pf["x_max"] - pf["x_min"]) if (pf["x_max"] > pf["x_min"]) else 1.0 # type: ignore
        y_deny = (pf["y_max"] - pf["y_min"]) if (pf["y_max"] > pf["y_min"]) else 1.0
        y_train = np.array([
            (y_true[0] - pf["x_min"]) / y_denx,
            (y_true[1] - pf["y_min"]) / y_deny
        ], dtype=np.float32)
        y_stats = np.array([pf["x_min"], pf["x_max"], pf["y_min"], pf["y_max"]], dtype=np.float32)
        return y_train, y_stats

    raise ValueError(f"Unknown y_norm_mode={y_norm_mode}")
   

def denorm_y(preds_norm, batch, y_norm_mode, stats=None, device=None):
    """
    Args:
        preds_norm: (B,2) torch
        batch: dict from dataloader, include y_raw and y_stats
        y_norm_mode: same string as dataset
        stats: train stats dict when using global_* modes
    Return: 
        preds_real: (B,2) torch in real coordinate system
    """
    if device is None:
        device = preds_norm.device

    if y_norm_mode == "none":
        return preds_norm

    if y_norm_mode == "per_file_minmax":
        assert "y_stats" in batch and batch["y_stats"] is not None
        # batch["y_stats"]: (B,4) = [x_min, x_max, y_min, y_max]
        s = batch["y_stats"].to(device).float()
        x_min, x_max, y_min, y_max = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        pred_x = preds_norm[:, 0] * (x_max - x_min) + x_min
        pred_y = preds_norm[:, 1] * (y_max - y_min) + y_min
        return torch.stack([pred_x, pred_y], dim=1)

    if y_norm_mode == "global_zscore":
        assert stats is not None, "global_zscore 需要 train stats"
        y_mean = torch.tensor(stats["y_mean"], device=device, dtype=torch.float32).view(1, 2)
        y_std  = torch.tensor(stats["y_std"],  device=device, dtype=torch.float32).view(1, 2)
        return preds_norm * (y_std + 1e-6) + y_mean

    if y_norm_mode == "global_minmax":
        assert stats is not None, "global_minmax 需要 train stats"
        y_min = torch.tensor(stats["y_min"], device=device, dtype=torch.float32).view(1, 2)
        y_max = torch.tensor(stats["y_max"], device=device, dtype=torch.float32).view(1, 2)
        return preds_norm * (y_max - y_min + 1e-6) + y_min

    raise ValueError(f"Unknown y_norm_mode: {y_norm_mode}")
    
def decompose_gyro_to_v1_v2_np(gyro):
    """
    EqNIO（TLIO）风格的稳定分解：
    将角速度 ω 分解为 (v1, v2)，使得 cross(v1, v2) ≈ ω。

    参数
    ----------
    gyro : np.ndarray
        形状 (T, 3)，float32，表示角速度序列。

    返回
    -------
    v1, v2 : np.ndarray
        形状 (T, 3)，float32。

    说明
    ----
    - 在数据已与重力方向对齐（EqNIO 的假设）时效果最好。
    - 对接近退化的情况提供了兜底处理（例如 ω 的 xy 分量接近 0）。
    - 将 v1 / v2 的模长缩放到与 ||gyro|| 一致，
      以提高训练稳定性（与 EqNIO 的做法一致）。
    """
    # 确保数据类型为 float32
    gyro = gyro.astype(np.float32)

    # 绕 Z 轴旋转 90° 的旋转矩阵
    Rz90 = np.array([[0.0, -1.0, 0.0],
                     [1.0,  0.0, 0.0],
                     [0.0,  0.0, 1.0]], dtype=np.float32)

    # 将 gyro 在 XY 平面内旋转 90°
    gyro_flip = (gyro @ Rz90.T).astype(np.float32)
    # 去掉 Z 分量，仅保留 XY 信息
    gyro_flip[:, 2] = 0.0

    # 第一层叉乘，构造 v1
    v1 = np.cross(gyro, gyro_flip).astype(np.float32)
    # 第二层叉乘，构造 v2
    v2 = np.cross(gyro, v1).astype(np.float32)

    # 计算 gyro 在 XY 平面的范数
    gyro_xy_norm = np.linalg.norm(gyro[:, :2], axis=1)
    # 标记接近退化的情况（XY 分量几乎为 0）
    mask = (gyro_xy_norm < 1e-8)
    if np.any(mask):
        # 使用固定的 y 轴方向作为辅助向量
        x = np.zeros_like(gyro, dtype=np.float32)
        x[:, 1] = 1.0  # y 轴
        v1[mask] = np.cross(x[mask], gyro[mask]).astype(np.float32)
        v2[mask] = np.cross(gyro[mask], v1[mask]).astype(np.float32)

    # 计算范数，用于后续缩放
    gyro_norm = np.linalg.norm(gyro, axis=1, keepdims=True).astype(np.float32)
    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True).astype(np.float32)
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True).astype(np.float32)

    # 将 v1 / v2 的模长缩放到与 gyro 的模长一致
    v1 = v1 * gyro_norm / np.clip(v1_norm, 1e-7, 1e13)
    v2 = v2 * gyro_norm / np.clip(v2_norm, 1e-7, 1e13)

    # 对几乎为零的角速度进行特殊处理，直接置零
    tiny = (gyro_norm[:, 0] < 1e-6)
    if np.any(tiny):
        v1[tiny] = 0.0
        v2[tiny] = 0.0

    return v1.astype(np.float32), v2.astype(np.float32)


def rodrigues_rot_from_a_to_b(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    求旋转矩阵 R，使得 R @ a = b（a,b 为 3D 向量，内部会单位化）。
    返回: (3,3) float32
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < eps:
        # 平行或反平行
        if c > 0:
            return np.eye(3, dtype=np.float32)
        # 180°：找一个与 a 不共线的轴
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + eps)
        return (-np.eye(3, dtype=np.float32) + 2.0 * np.outer(v, v)).astype(np.float32)

    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]], dtype=np.float32)

    R = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s * s + eps))
    return R.astype(np.float32)


def gravity_align_per_window(
    mag_win: np.ndarray,
    acc_win: np.ndarray,
    gyro_win: np.ndarray,
    grav_win: np.ndarray,
    *,
    z_axis: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
    eps: float = 1e-8,
    use_linear_acc: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 Android TYPE_GRAVITY 做 per-window 重力对齐（更稳）。

    输入:
      mag_win/acc_win/gyro_win/grav_win: (L,3)
      use_linear_acc:
        - True: 先做 a_lin = accel - gravity，再旋转
        - False: 直接旋转 accel（含重力）

    输出:
      对齐后的 (mag, acc, gyro, gravity) 窗口，shape 仍为 (L,3)
    """
    # 用窗口内 gravity 的均值作为重力方向（更平滑、更稳）
    g_mean = grav_win.mean(axis=0).astype(np.float32)
    g_norm = np.linalg.norm(g_mean)
    if g_norm < eps:
        # 极端情况：gravity 全 0，直接返回原数据
        return mag_win, acc_win, gyro_win, grav_win

    Rg = rodrigues_rot_from_a_to_b(g_mean, z_axis, eps=eps)  # body -> gravity-aligned

    if use_linear_acc:
        acc_in = (acc_win - grav_win).astype(np.float32)
    else:
        acc_in = acc_win.astype(np.float32)

    # 行向量右乘：x' = x @ R^T
    mag_out  = (mag_win.astype(np.float32)  @ Rg.T).astype(np.float32)
    acc_out  = (acc_in @ Rg.T).astype(np.float32)
    gyro_out = (gyro_win.astype(np.float32) @ Rg.T).astype(np.float32)
    grav_out = (grav_win.astype(np.float32) @ Rg.T).astype(np.float32)

    return mag_out, acc_out, gyro_out, grav_out