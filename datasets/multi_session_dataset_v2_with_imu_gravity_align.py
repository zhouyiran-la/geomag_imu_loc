import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from .utils import compute_train_stats_from_csv_files, norm_y, decompose_gyro_to_v1_v2_np, rodrigues_rot_from_a_to_b, gravity_align_per_window

MAG_COLS = ["geomagneticx", "geomagneticy", "geomagneticz"]
POS_COLS = ["pos_x", "pos_y"]
ACC_COLS = ["accX", "accY", "accZ"]          # <-- change to your CSV column names if needed
GYRO_COLS = ["gyroX", "gyroY", "gyroZ"]            # <-- change to your CSV column names if needed
GRAV_COLS = ["gravityX", "gravityY", "gravityY"]   # <-- Android TYPE_GRAVITY columns (assumed)

class MagneticImuDataSet(Dataset):
    """MagneticDataSetV2:

    - 从同一个 CSV 文件中读取磁力计 + IMU 数据（加速度 acc、陀螺仪 gyro、重力 grav）
    - 构造滑动时间窗口
    - 从陀螺仪数据中生成 EqNIO 风格的 v1 / v2 分解，
      用于 O(2) FrameNet
    - （新增）使用 Android 的 TYPE_GRAVITY，对每个时间窗口进行重力对齐

    Returned sample:
      {
        "x_mag": (L,3),
        "x_acc": (L,3),
        "x_gyro": (L,3),
        "x_grav": (L,3),
        "x_v1": (L,3),
        "x_v2": (L,3),
        "y": (2,), "y_raw": (2,), 
        "fid": int, 
        "y_stats": dict
      }
    """

    def __init__(
        self,
        file_paths,
        *,
        seq_len=128,
        stride=1,
        stats=None,
        normalize_mag=False,
        normalize_imu=False,
        y_norm_mode="per_file_minmax",
        transform=None,
        cache_in_memory=True,
        gravity_align=True,
        use_linear_acc=False,
    ):
        self.file_paths = [str(p) for p in file_paths]
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.stats = stats
        self.normalize_mag = bool(normalize_mag)
        self.normalize_imu = bool(normalize_imu)
        self.y_norm_mode = y_norm_mode
        self.transform = transform
        self.cache_in_memory = bool(cache_in_memory)

        # NEW
        self.gravity_align = bool(gravity_align)
        self.use_linear_acc = bool(use_linear_acc)

        if (self.normalize_mag or self.normalize_imu) and self.stats is None:
            raise ValueError("normalize_mag/normalize_imu=True requires stats (means/stds).")

        if self.y_norm_mode in ("global_zscore", "global_minmax") and self.stats is None:
            raise ValueError("global y normalization requires stats (y_*).")

        self._cache = {}
        self.index = []
        self.lengths = []

        for fid, p in enumerate(self.file_paths):
            T = self._peek_length(p)
            self.lengths.append(T)
            if T >= self.seq_len:
                for s in range(0, T - self.seq_len + 1, self.stride):
                    self.index.append((fid, s))

    def _peek_length(self, path):
        df = pd.read_csv(path, usecols=[MAG_COLS[0]])
        return len(df)

    def _pos_minmax_stats(self, pos):
        x = pos[:, 0].astype(np.float32)
        y = pos[:, 1].astype(np.float32)
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def _load_one_file(self, fid):
        if self.cache_in_memory and fid in self._cache:
            return self._cache[fid]

        p = self.file_paths[fid]
        df = pd.read_csv(p)
        print(f"已读取{p}，length={len(df)}")

        missing = [c for c in (MAG_COLS + POS_COLS + ACC_COLS + GYRO_COLS + GRAV_COLS) if c not in df.columns]
        if missing:
            raise KeyError(
                f"CSV missing columns: {missing}. "
                f"Please update MAG_COLS/POS_COLS/ACC_COLS/GYRO_COLS/GRAV_COLS to match your file."
            )

        x_mag = df[MAG_COLS].to_numpy(dtype=np.float32)       # (T,3)
        x_acc = df[ACC_COLS].to_numpy(dtype=np.float32)       # (T,3)
        x_gyro = df[GYRO_COLS].to_numpy(dtype=np.float32)     # (T,3)
        x_grav = df[GRAV_COLS].to_numpy(dtype=np.float32)     # (T,3)
        y_raw = df[POS_COLS].to_numpy(dtype=np.float32)       # (T,2)

        y_pf_stats = self._pos_minmax_stats(y_raw)

        data = {
            "x_mag": x_mag,
            "x_acc": x_acc,
            "x_gyro": x_gyro,
            "x_grav": x_grav,
            "y_raw": y_raw,
            "y_pf_stats": y_pf_stats
        }
        if self.cache_in_memory:
            self._cache[fid] = data
        return data

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fid, s = self.index[idx]
        data = self._load_one_file(fid)

        e = s + self.seq_len
        mag_win  = data["x_mag"][s:e]
        acc_win  = data["x_acc"][s:e]
        gyro_win = data["x_gyro"][s:e]
        grav_win = data["x_grav"][s:e]
        y_true = data["y_raw"][e - 1]
        pf = data["y_pf_stats"]

        # 重力对齐（per-window）：先对齐，再做后续归一化 / v1v2
        if self.gravity_align:
            mag_win, acc_win, gyro_win, grav_win = gravity_align_per_window(
                mag_win, acc_win, gyro_win, grav_win,
                use_linear_acc=self.use_linear_acc
            )

        # 可选输入归一化（注意：若开启 gravity_align，建议统计量 stats 也在对齐后数据上计算）
        if self.normalize_mag:
            assert self.stats
            mu = self.stats["mag_mean"][None, :]
            sd = self.stats["mag_std"][None, :]
            mag_win = (mag_win - mu) / (sd + 1e-6)

        if self.normalize_imu:
            assert self.stats
            imu = np.concatenate([acc_win, gyro_win], axis=1)
            mu = self.stats["imu_mean"][None, :]
            sd = self.stats["imu_std"][None, :]
            imu = (imu - mu) / (sd + 1e-6)
            acc_win, gyro_win = imu[:, :3].astype(np.float32), imu[:, 3:6].astype(np.float32)

        # EqNIO-style ω -> (v1, v2) 分解（用对齐后的 gyro）
        v1_win, v2_win = decompose_gyro_to_v1_v2_np(gyro_win)

        # 标签归一化
        y_train, y_stats = norm_y(self.y_norm_mode, y_true, pf, self.stats)

        sample = {
            "x_mag":  mag_win.astype(np.float32),
            "x_acc":  acc_win.astype(np.float32),
            "x_gyro": gyro_win.astype(np.float32),
            "x_v1":   v1_win.astype(np.float32),
            "x_v2":   v2_win.astype(np.float32),
            # 可选：保留对齐后的重力向量，便于 debug/可视化
            "x_grav": grav_win.astype(np.float32),
            "y": y_train.astype(np.float32),
            "y_raw": y_true.astype(np.float32),
            "fid": fid,
            "y_stats": y_stats,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def create_magnetic_imu_dataset_dataloader(
    data_dir,
    batch_size=64,
    pattern=".csv",
    num_workers=0,
    shuffle_train=True,
    pin_memory=False,
    transform=None,
    stats=None,
    *,
    seq_len=128,
    stride=1,
    normalize_mag=False,
    normalize_imu=False,
    y_norm_mode="per_file_minmax",
    cache_in_memory=True,
    gravity_align=True,
    use_linear_acc=False,
):
    def list_files(file_dir: str):
        if not os.path.isdir(file_dir):
            print(f"目录不存在，跳过: {file_dir}")
            return None
        files = sorted([str(p) for p in Path(file_dir).glob(f"*{pattern}")])
        if len(files) == 0:
            print(f"目录下无匹配文件，跳过: {file_dir}")
            return None
        return files

    files = list_files(data_dir)
    if files is None:
        return None

    dataset = MagneticImuDataSet(
        files,
        seq_len=seq_len,
        stride=stride,
        stats=stats,
        normalize_mag=normalize_mag,
        normalize_imu=normalize_imu,
        y_norm_mode=y_norm_mode,
        transform=transform,
        cache_in_memory=cache_in_memory,
        gravity_align=gravity_align,
        use_linear_acc=use_linear_acc,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
