import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1️⃣ Default Transform
# ============================================================

class DefaultTransform:
    """numpy → tensor 转换，并尽量保留原有字段。"""
    def __call__(self, sample):
        # magnetometer
        if "x_mag" in sample and not isinstance(sample["x_mag"], torch.Tensor):
            sample["x_mag"] = torch.tensor(sample["x_mag"], dtype=torch.float32)

        # IMU (optional)
        for k in ["x_acc", "x_gyro", "x_v1", "x_v2"]:
            if k in sample and not isinstance(sample[k], torch.Tensor):
                sample[k] = torch.tensor(sample[k], dtype=torch.float32)

        # targets / metadata
        if "y" in sample and not isinstance(sample["y"], torch.Tensor):
            sample["y"] = torch.tensor(sample["y"], dtype=torch.float32)
        if "y_raw" in sample and not isinstance(sample["y_raw"], torch.Tensor):
            sample["y_raw"] = torch.tensor(sample["y_raw"], dtype=torch.float32)
        if "fid" in sample and not isinstance(sample["fid"], torch.Tensor):
            sample["fid"] = torch.tensor(sample["fid"], dtype=torch.int32)

        # y_stats is often a dict; keep it as-is unless you explicitly want tensorization
        return sample


# ============================================================
# 2️⃣ 论文特征增强（FeatureAugmentTransform）
# ============================================================

class FeatureAugmentTransform:
    """
    二次滑窗 + 邻域特征增强 Transform
    输入: x_mag (W₁, 3)
    输出: x_mag_aug (W₁−W₂+1, W₂, 3×W₂)
    """

    def __init__(self, W2: int = 5):
        self.W2 = W2

    def __call__(self, sample):
        x_mag = sample["x_mag"]
        if isinstance(x_mag, torch.Tensor):
            x_mag = x_mag.numpy()

        W1, C = x_mag.shape
        W2 = self.W2
        assert C == 3, f"地磁输入应为三维 (m_s, m_h, m_v)，但得到 {C} 维"
        assert W2 <= W1, f"W2 必须小于等于 W1 (当前: W2={W2}, W1={W1})"

        local_features = []
        for start in range(W1 - W2 + 1):
            window = x_mag[start:start+W2]
            neighborhood = []
            for j in range(W2):
                rotated = np.roll(window, -j, axis=0).reshape(-1)
                neighborhood.append(rotated)
            neighborhood = np.stack(neighborhood, axis=0)
            local_features.append(neighborhood)

        x_mag_aug = np.stack(local_features, axis=0)
        sample["x_mag_aug"] = torch.from_numpy(x_mag_aug).float()
        return sample


# ============================================================
# 3️⃣ 地磁梯度特征增强
# ============================================================

class MagneticGradientTransform:
    """
    添加地磁梯度与二阶差分特征：
    输入: x_mag (W₁, 3)
    输出: x_mag_grad (W₁−2, 9)
    """
    def __call__(self, sample):
        x_mag = sample["x_mag"]
        if isinstance(x_mag, torch.Tensor):
            x_mag = x_mag.numpy()

        delta = np.diff(x_mag, axis=0)
        delta2 = np.diff(delta, axis=0)
        x_base = x_mag[2:]
        x_aug = np.concatenate([x_base, delta[1:], delta2], axis=-1)
        sample["x_mag_grad"] = torch.tensor(x_aug, dtype=torch.float32)
        return sample


# ============================================================
# 4️⃣ 频谱特征增强
# ============================================================

class MagneticSpectralTransform:
    """
    计算地磁信号的短时傅里叶谱特征（STFT）
    输出: x_mag_spectral
    """
    def __init__(self, n_fft=64, hop_length=16):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, sample):
        x_mag = sample["x_mag"]
        if isinstance(x_mag, np.ndarray):
            x_mag = torch.tensor(x_mag, dtype=torch.float32)
        x_mag = x_mag.transpose(0, 1)  # (3, W₁)

        specs = []
        for i in range(3):
            spec = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft, hop_length=self.hop_length
            )(x_mag[i])
            spec = torch.log1p(spec)
            spec_mean = torch.mean(spec, dim=0)
            spec_max = torch.max(spec, dim=0).values
            spec_vec = torch.cat([spec_mean, spec_max], dim=0)
            specs.append(spec_vec)
        sample["x_mag_spectral"] = torch.stack(specs, dim=0).T  # (time, feature)
        return sample

# ============================================================
#  5️⃣ Yaw / O(2) augmentation for EqNIO-style canonical consistency
# ============================================================

class YawAugmentO2Transform:
    """
    随机对磁力计 + IMU的XY分量施加O(2)变换
    （包括偏航旋转 yaw + 可选镜像反射）。
    该增强用于构造一对样本，以支持 canonical consistency loss。

    原输入（torch.Tensor 或 numpy.ndarray）：
        x_mag: (T, 3)
        x_acc: (T, 3)
        x_v1:  (T, 3)
        x_v2:  (T, 3)

    新增：
        aug: dict，包含增强后的张量（键与原始一致）
        o2_R: (2, 2) 张量，作用于 XY 平面的 O(2) 变换矩阵
        o2_is_reflect: bool，是否包含反射
    """
    def __init__(self, p_reflect: float = 0.5,
                 angle_range=(0.0, 2 * 3.141592653589793)):
        # 发生反射（镜像）的概率
        self.p_reflect = float(p_reflect)
        # 旋转角度采样范围（弧度）
        self.angle_range = angle_range

    @staticmethod
    def _apply_R_xy(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        对输入张量的 XY 分量应用 2×2 线性变换矩阵 R，
        Z 分量保持不变。

        参数：
            x: (T, 3)
            R: (2, 2)

        返回：
            (T, 3) 变换后的张量
        """
        xy = x[:, :2]          # 取 XY 分量 (T, 2)
        xy2 = xy @ R.T         # 右乘变换矩阵
        return torch.cat([xy2, x[:, 2:3]], dim=-1)

    def __call__(self, sample):
        """
        对一个样本执行随机 O(2) 偏航增强。

        参数：
            sample: dict，包含原始传感器数据

        返回：
            sample: 增强后的样本（原地修改并返回）
        """
        # 确保相关输入都是 torch.Tensor
        for k in ["x_mag", "x_acc", "x_v1", "x_v2", "x_gyro"]:
            if k in sample and not isinstance(sample[k], torch.Tensor):
                sample[k] = torch.tensor(sample[k], dtype=torch.float32)

        # 在给定范围内随机采样一个偏航角
        a0, a1 = self.angle_range
        theta = (a0 + (a1 - a0) * torch.rand((), dtype=torch.float32)).item()

        # 构造二维旋转矩阵
        c = float(torch.cos(torch.tensor(theta)))
        s = float(torch.sin(torch.tensor(theta)))
        R = torch.tensor([[c, -s],
                          [s,  c]], dtype=torch.float32)

        # 可选的反射操作（关于 y 轴镜像）
        is_reflect = bool((torch.rand(()) < self.p_reflect).item())
        if is_reflect:
            S = torch.tensor([[1.0, 0.0],
                              [0.0, -1.0]], dtype=torch.float32)
            R = R @ S

        # 对各个向量通道应用 XY 平面变换
        aug = {}
        for k in ["x_mag", "x_acc", "x_v1", "x_v2"]:
            if k in sample:
                aug[k] = self._apply_R_xy(sample[k], R)

        # （可选）陀螺仪：作为 3D 向量，仅用于数据增强
        if "x_gyro" in sample:
            aug["x_gyro"] = self._apply_R_xy(sample["x_gyro"], R)

        # 保存增强结果及对应的 O(2) 变换信息
        sample["aug"] = aug
        sample["o2_R"] = R
        sample["o2_is_reflect"] = is_reflect
        return sample


# ============================================================
# 6️⃣组合多种 Transform
# ============================================================

class ComposeTransform:
    """顺序执行多个 transform"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample