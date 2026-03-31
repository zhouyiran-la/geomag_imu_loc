import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

from plot.utils.plot_style import setup_plot_equal_style, save_figure

setup_plot_equal_style()

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "12-25-信息文管室内地磁数据采集"/"12-25-MEIZU 20"/"12-25-信息"/"dataset_2025-12-25_20-45-32-041.csv"

ACC_FILTER_OUTPUT_PATH = ROOT / "figures" / "imu_acc_lowpass_compare.png"
GYRO_FILTER_OUTPUT_PATH = ROOT / "figures" / "imu_gyro_lowpass_compare.png"

# =========================
# 参数
# =========================
START_INDEX = 5000
MAX_SAMPLES = 1000

FIG_SIZE = (6.0, 5.0)

# 低通滤波系数，越小越平滑
ALPHA_ACC = 0.30
ALPHA_GYRO = 0.30

RAW_LINE_WIDTH = 1.5
FILT_LINE_WIDTH = 1.5

RAW_ALPHA = 0.45
FILT_ALPHA = 1.0

# =========================
# 读取数据
# =========================
df = pd.read_csv(DATA_PATH)
df = df.iloc[START_INDEX: START_INDEX + MAX_SAMPLES].copy()

imu_columns = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]
for col in imu_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=imu_columns).reset_index(drop=True)  # type: ignore

x = np.arange(len(df))

accX = df["accX"].to_numpy()
accY = df["accY"].to_numpy()
accZ = df["accZ"].to_numpy()

gyroX = df["gyroX"].to_numpy()
gyroY = df["gyroY"].to_numpy()
gyroZ = df["gyroZ"].to_numpy()


# =========================
# 一阶IIR低通滤波
# y[t] = alpha * x[t] + (1-alpha) * y[t-1]
# =========================
def lowpass_filter(signal: np.ndarray, alpha: float) -> np.ndarray:
    signal = signal.astype(np.float32)
    filtered = np.zeros_like(signal, dtype=np.float32)

    if len(signal) == 0:
        return filtered

    filtered[0] = signal[0]
    for i in range(1, len(signal)):
        filtered[i] = alpha * signal[i] + (1.0 - alpha) * filtered[i - 1]

    return filtered


accX_filt = lowpass_filter(accX, ALPHA_ACC)
accY_filt = lowpass_filter(accY, ALPHA_ACC)
accZ_filt = lowpass_filter(accZ, ALPHA_ACC)

gyroX_filt = lowpass_filter(gyroX, ALPHA_GYRO)
gyroY_filt = lowpass_filter(gyroY, ALPHA_GYRO)
gyroZ_filt = lowpass_filter(gyroZ, ALPHA_GYRO)


# =========================
# 绘图函数：原始 vs 滤波后
# =========================
def plot_filtered_comparison(
    x,
    raw_data,
    filtered_data,
    labels,
    raw_color,
    filtered_color,
    output_path,
    figsize=(8.5, 6.0),
):
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for i in range(3):
        ax = axes[i]

        y_raw = raw_data[i]
        y_filt = filtered_data[i]

        # 原始信号
        ax.plot(
            x,
            y_raw,
            color=raw_color,
            linewidth=RAW_LINE_WIDTH,
            alpha=RAW_ALPHA,
        )

        # 滤波后信号
        ax.plot(
            x,
            y_filt,
            color=filtered_color,
            linewidth=FILT_LINE_WIDTH,
            alpha=FILT_ALPHA,
        )

        ax.set_ylabel(labels[i])

        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            length=4,
            width=1.0,
        )

        # 根据量级自动调整显示精度
        max_abs_val = max(np.max(np.abs(y_raw)), np.max(np.abs(y_filt)))
        if max_abs_val < 0.05:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.grid(False)
        # ax.legend(loc="upper right", frameon=False)

    axes[-1].set_xlabel("样本序号")

    axes[-1].set_xlim(x[0]-15, x[-1]+15)

    fig.subplots_adjust(
        left=0.12,
        right=0.96,
        bottom=0.08,
        top=0.92,
        hspace=0.05,
    )

    fig.legend(
        ["滤波前", "滤波后"],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=12,
        columnspacing=3.0
    )

    save_figure(fig, output_path, show=False, tight=False)
    plt.close(fig)


# =========================
# 加速度三轴滤波对比图
# =========================
plot_filtered_comparison(
    x=x,
    raw_data=[accX, accY, accZ],
    filtered_data=[accX_filt, accY_filt, accZ_filt],
    labels=["accX(m/s²)", "accY(m/s²)", "accZ(m/s²)"],
    raw_color="#7bc2e8",
    filtered_color="#1f77b4",
    output_path=ACC_FILTER_OUTPUT_PATH,
    figsize=FIG_SIZE,
)

# =========================
# 陀螺仪三轴滤波对比图
# =========================
plot_filtered_comparison(
    x=x,
    raw_data=[gyroX, gyroY, gyroZ],
    filtered_data=[gyroX_filt, gyroY_filt, gyroZ_filt],
    labels=["gyroX(rad/s)", "gyroY(rad/s)", "gyroZ(rad/s)"],
    raw_color="#fcb36a",
    filtered_color="#ff7f0e",
    output_path=GYRO_FILTER_OUTPUT_PATH,
    figsize=FIG_SIZE,
)

print("绘图完成：")
print(ACC_FILTER_OUTPUT_PATH)
print(GYRO_FILTER_OUTPUT_PATH)