import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

from plot.utils.plot_style import setup_plot_equal_style, style_axis, save_figure

setup_plot_equal_style()

ROOT = Path(__file__).resolve().parents[1]

# DATA_PATH = ROOT / "data" / "12-25-信息文管室内地磁数据采集"/"12-25-MEIZU 20"/"12-25-文管"/"dataset_2025-12-25_19-18-16-132.csv"
DATA_PATH = ROOT / "data" / "12-25-信息文管室内地磁数据采集"/"12-25-MEIZU 20"/"12-25-信息"/"dataset_2025-12-25_20-45-32-041.csv"
# DATA_PATH = ROOT / "data" / "12-25-信息文管室内地磁数据采集"/"12-25-Huawei P60"/"12-25-信息"/"dataset_2025-12-25_20-45-06-309.csv"
# DATA_PATH = ROOT / "data" / "12-25-信息文管室内地磁数据采集"/"12-25-OPPO Find X"/"12-25-信息"/"dataset_2025-12-25_20-45-06-123.csv"
# DATA_PATH = ROOT / "data" / "12-25-信息文管室内地磁数据采集"/"12-25-MEIZU 20"/"12-25-信息"/"dataset_2025-12-25_20-49-53-750.csv"


ACC_FFT_OUTPUT_PATH = ROOT / "figures" / "imu_acc_norm_fft_meizu_xinxi_orange.png"
GYRO_FFT_OUTPUT_PATH = ROOT / "figures" / "imu_gyro_norm_fft_honor_xinxi.png"

# =========================
# 参数控制
# =========================
START_INDEX = 0
MAX_SAMPLES = 8192      # FFT建议取2的幂附近，当然不是必须
FS = 100.0              # IMU采样率(Hz)，这里请改成你的真实采样率
FIG_SIZE = (8, 5)

# 若只关注低频步态范围，可限制显示频率上限
FREQ_MAX = 100         # Hz

# =========================
# 读取数据
# =========================
df = pd.read_csv(DATA_PATH)
df = df.iloc[START_INDEX: START_INDEX + MAX_SAMPLES]

x = np.arange(len(df))

accX = df["accX"].to_numpy()
accY = df["accY"].to_numpy()
accZ = df["accZ"].to_numpy()

gyroX = df["gyroX"].to_numpy()
gyroY = df["gyroY"].to_numpy()
gyroZ = df["gyroZ"].to_numpy()

# =========================
# 计算模长
# =========================
acc_norm = np.sqrt(accX**2 + accY**2 + accZ**2)
gyro_norm = np.sqrt(gyroX**2 + gyroY**2 + gyroZ**2)

# =========================
# FFT函数
# =========================
def compute_single_sided_fft(signal, fs):
    """
    返回:
        freqs: 单边频率轴
        amp:   单边幅值谱
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    # 去均值，减弱直流分量
    signal = signal - np.mean(signal)

    # 加窗（Hanning窗），减小频谱泄漏
    window = np.hanning(n)
    signal_win = signal * window

    fft_vals = np.fft.rfft(signal_win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # 幅值归一化
    amp = np.abs(fft_vals) * 2.0 / np.sum(window)

    return freqs, amp


def plot_fft_spectrum(
    signal,
    fs,
    output_path,
    color,
    ylabel="Amplitude",
    xlabel="Frequency (Hz)",
    figsize=(8.5, 3.8),
    freq_max=None,
):
    freqs, amp = compute_single_sided_fft(signal, fs)

    if freq_max is not None:
        mask = freqs <= freq_max
        freqs = freqs[mask]
        amp = amp[mask]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(freqs, amp, color=color)

    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)

    ax.grid(False)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5,
        width=1.0,
    )

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # 如果可用，也可以打开
    # style_axis(ax)

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.18,
        top=0.96,
    )

    save_figure(fig, output_path, show=False, tight=False)
    plt.close(fig)


# =========================
# 绘制加速度模长FFT
# =========================
plot_fft_spectrum(
    signal=acc_norm,
    fs=FS,
    output_path=ACC_FFT_OUTPUT_PATH,
    color="#F18C54",
    ylabel="幅值",
    xlabel = "频率 (Hz)",
    figsize=FIG_SIZE,
    freq_max=FREQ_MAX,
)

# # =========================
# # 绘制陀螺仪模长FFT
# # =========================
# plot_fft_spectrum(
#     signal=gyro_norm,
#     fs=FS,
#     output_path=GYRO_FFT_OUTPUT_PATH,
#     color="orange",
#     ylabel="幅值",
#     xlabel = "频率 (Hz)",
#     figsize=FIG_SIZE,
#     freq_max=FREQ_MAX,
# )