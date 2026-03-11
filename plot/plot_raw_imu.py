import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patheffects as pe
from matplotlib.ticker import FormatStrFormatter
from plot.utils.plot_style import setup_plot_equal_style, style_axis, save_figure

setup_plot_equal_style()

ROOT = Path(__file__).resolve().parents[1]
# DATA_PATH = ROOT / "data" / "12-25-信息文管室内地磁数据采集"/"12-25-Honor 200"/"12-25-信息"/"TZ"/"data_with_label_dataset_2025-12-25_20-45-10-018_T_Z.csv"
# DATA_PATH = ROOT / "data" / "12-25-信息文管室内地磁数据采集"/"12-25-Xiaomi 14"/"12-25-信息"/"dataset_2025-12-25_20-45-10-523.csv"
# DATA_PATH = ROOT / "data" / "12-25-信息文管室内地磁数据采集"/"12-25-MEIZU 20"/"12-25-文管"/"dataset_2025-12-25_19-18-16-132.csv"
DATA_PATH = ROOT / "data" / "12-25-信息文管室内地磁数据采集"/"12-25-MEIZU 20"/"12-25-信息"/"dataset_2025-12-25_20-45-32-041.csv"
ACC_OUTPUT_PATH = ROOT / "figures" / "imu_acc_meizu_xinxi.png"
GYRO_OUTPUT_PATH = ROOT / "figures" / "imu_gyro_meizu_xinxi.png"
IMU_OUTPUT_PATH = ROOT / "figures" / "imu_acc_gyro_meizu_xinxi.png"
# =========================
# 参数控制
# =========================
START_INDEX = 5000
MAX_SAMPLES = 1000

# 子图排版参数
FIG_SIZE = (9.5, 6.5)
HSPACE = 0.03
WSPACE = 0.18
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

# colors = ["#207ECA", "#FDADD4", "#FDE56F"]
# colors = ["#1f77b4", "#FDADD4", "#FDE56F"]
def symmetric_ylim(data, margin=0.1):
    """
    根据数据自动生成对称的 y 轴范围
    """
    max_val = np.max(np.abs(data))
    limit = (1 + margin) * max_val
    return -limit, limit

def plot_two_column_imu(
    x,
    accX,
    accY,
    accZ,
    gyroX,
    gyroY,
    gyroZ,
    output_path,
    xlabel="样本序号",
    acc_color="#1f77b4",
    gyro_color="orange",
    figsize=(10, 7),
    hspace=0.05,
    wspace=0.15,
):

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)

    acc_series = [accX, accY, accZ]
    gyro_series = [gyroX, gyroY, gyroZ]

    acc_labels = ["accX", "accY", "accZ"]
    gyro_labels = ["gyroX", "gyroY", "gyroZ"]

    for i in range(3):

        # 左列：加速度
        ax_acc = axes[i, 0]
        ax_acc.plot(x, acc_series[i], color=acc_color)
        ax_acc.set_ylabel(acc_labels[i], labelpad=6)
        ax_acc.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax_acc.grid(False)
        # ymin, ymax = symmetric_ylim(acc_series[i])
        # ax_acc.set_ylim(ymin, ymax)
        ax_acc.tick_params(
            axis="y",
            which="major",
            direction="in",
            length=5,
            width=1.0,
        )

        # 右列：陀螺仪
        ax_gyro = axes[i, 1]
        ax_gyro.plot(x, gyro_series[i], color=gyro_color)
        ax_gyro.set_ylabel(gyro_labels[i], labelpad=6)
        ax_gyro.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax_gyro.grid(False)
        ymin, ymax = symmetric_ylim(gyro_series[i])
        ax_gyro.set_ylim(ymin, ymax)
        ax_gyro.tick_params(
            axis="y",
            which="major",
            direction="in",
            length=5,
            width=1.0,
        )

    # 最底部加 x 轴
    axes[2, 0].set_xlabel(xlabel, labelpad=0)
    axes[2, 1].set_xlabel(xlabel, labelpad=0)

    # 顶部标题
    axes[0, 0].set_title("加速度计", pad=6)
    axes[0, 1].set_title("陀螺仪", pad=6)

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.08,
        top=0.96,
        hspace=hspace,
        wspace=wspace,
    )

    save_figure(fig, output_path, show=False, tight=False)
    plt.close(fig)

def plot_three_axis_signal(
    x,
    y1,
    y2,
    y3,
    ylabels,
    output_path,
    xlabel="样本序号",
    color="blue",
    figsize=(8.5, 7.5),
    hspace=0.32,
):
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    series = [y1, y2, y3]

    for i, ax in enumerate(axes):
        ax.plot(x, series[i], color=color)
        # ax.plot(x, series[i])
        ax.set_ylabel(ylabels[i], labelpad=6)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.grid(False)
        if i < 2:
         ax.tick_params(axis='y',which="major",direction="in",length=5,width=1.0,)
        else:
            ax.tick_params(
                axis="both",
                which="major",
                direction="in",
                length=5,
                width=1.0,
            )

        # 如果你封装的 style_axis 可用，也可以打开
        # style_axis(ax, title:N, xlabel: str = "Time Step", ylabel: str = "Value")

    # 只有最底部子图显示 x 轴标签
    axes[-1].set_xlabel(xlabel, labelpad=6)

    # # 去掉上面两个子图重复的 x 轴刻度文字
    # for ax in axes[:-1]:
    #     ax.tick_params(axis='y')

    # 手动控制边距和子图间距
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.08,
        top=0.98,
        hspace=hspace,
    )

    save_figure(fig, output_path, show=False, tight=False)
    plt.close(fig)


# # =========================
# # 加速度图
# # =========================
# plot_three_axis_signal(
#     x=x,
#     y1=accX,
#     y2=accY,
#     y3=accZ,
#     ylabels=["accX", "accY", "accZ"],
#     color="#1f77b4",
#     output_path=ACC_OUTPUT_PATH,
#     figsize=FIG_SIZE,
#     hspace=HSPACE,
# )

# # =========================
# # 角速度图
# # =========================
# plot_three_axis_signal(
#     x=x,
#     y1=gyroX,
#     y2=gyroY,
#     y3=gyroZ,
#     ylabels=["gyroX", "gyroY", "gyroZ"],
#     color="orange",
#     output_path=GYRO_OUTPUT_PATH,
#     figsize=FIG_SIZE,
#     hspace=HSPACE,
# )

plot_two_column_imu(
    x=x,
    accX=accX,
    accY=accY,
    accZ=accZ,
    gyroX=gyroX,
    gyroY=gyroY,
    gyroZ=gyroZ,
    output_path=IMU_OUTPUT_PATH,
    hspace=HSPACE,
    wspace=WSPACE
)