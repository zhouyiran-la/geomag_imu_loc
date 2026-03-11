import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

from plot.utils.plot_style import setup_plot_equal_style, style_axis, save_figure

setup_plot_equal_style()

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "12-25-信息文管室内地磁数据采集"/"12-25-MEIZU 20"/"12-25-文管"/"dataset_2025-12-25_19-18-16-132.csv"

ACC_NORM_OUTPUT_PATH = ROOT / "figures" / "imu_acc_norm_.png"
GYRO_NORM_OUTPUT_PATH = ROOT / "figures" / "imu_gyro_norm_meizu.png"

# =========================
# 参数控制
# =========================
START_INDEX = 5000
MAX_SAMPLES = 1000

FIG_SIZE = (8.5, 3.8)
LINE_WIDTH = 2.0

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
# 配色
# =========================
ACC_COLOR = "#38C1F3"
GYRO_COLOR = "#FF0000"


def plot_signal_norm(
    x,
    y,
    ylabel,
    output_path,
    color,
    xlabel="Sample Index",
    figsize=(8.5, 3.8),
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(x, y, color=color)

    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=10)

    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5,
        width=1.0,
    )

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # 如果你封装的 style_axis 可用，可以打开
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
# 加速度模长图
# =========================
plot_signal_norm(
    x=x,
    y=acc_norm,
    ylabel="|acc|",
    output_path=ACC_NORM_OUTPUT_PATH,
    color=ACC_COLOR,
    figsize=FIG_SIZE,
)

# =========================
# 陀螺仪模长图
# =========================
plot_signal_norm(
    x=x,
    y=gyro_norm,
    ylabel="|gyro|",
    output_path=GYRO_NORM_OUTPUT_PATH,
    color=GYRO_COLOR,
    figsize=FIG_SIZE,
)