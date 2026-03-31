import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

from plot.utils.plot_style import setup_plot_equal_style, save_figure

setup_plot_equal_style()

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "zero-bias" / "huawei_dataset_2026-03-30_15-55-18-671.csv"
# DATA_PATH = ROOT / "data" / "zero-bias" / "meizu_dataset_2026-03-30_15-52-33-006.csv"
# DATA_PATH = ROOT / "data" / "zero-bias" / "oppo_dataset_2026-03-31_19-58-30-812.csv"
ACC_OUTPUT_PATH = ROOT / "figures" / "imu_static_acc_3axis.png"
GYRO_OUTPUT_PATH = ROOT / "figures" / "imu_static_gyro_3axis.png"

# =========================
# 参数
# =========================
START_INDEX = 500 #  xiaomi huawei_500 oppo 800
MAX_SAMPLES = 1000

FIG_SIZE = (6.0, 5.0)

# =========================
# 读取数据
# =========================
df = pd.read_csv(DATA_PATH)
df = df.iloc[START_INDEX: START_INDEX + MAX_SAMPLES].copy()

imu_columns = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]
for col in imu_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=imu_columns).reset_index(drop=True) # type: ignore

x = np.arange(len(df))
accX = df["accX"].to_numpy()
accY = df["accY"].to_numpy()
accZ = df["accZ"].to_numpy()

gyroX = df["gyroX"].to_numpy()
gyroY = df["gyroY"].to_numpy()
gyroZ = df["gyroZ"].to_numpy()

def format_mean(mean_val):
    if abs(mean_val) < 1e-2:
        return f"均值：{mean_val:.3e}"   # 小量用科学计数法
    else:
        return f"均值：{mean_val:.3f}"   # 大量正常显示

# =========================
# 绘图函数（3轴垂直）
# =========================
def plot_3axis_vertical(
    x,
    data,
    labels,
    line_color,
    output_path,
    figsize=(8.5, 6),
):
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for i in range(3):
        ax = axes[i]
        y = data[i]

        mean_val = np.mean(y)
        print(format_mean(mean_val))

        # 主曲线
        ax.plot(x, y, color = line_color)

        # 均值线（带图例）
        ax.axhline(
            mean_val,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            color = "black",
            label=format_mean(mean_val),
        )

        # ✅ y轴对称（关键改动）
        max_dev = np.max(np.abs(y - mean_val))
        margin = max_dev * 1.2
        
        ax.set_ylim(mean_val - margin, mean_val + margin)

        ax.set_ylabel(labels[i])

        ax.grid(False)

        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            length=4,
            width=1.0,
        )

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        ax.legend(loc="upper right", frameon=False)

    axes[-1].set_xlabel("样本序号")
    axes[-1].set_xlim(x[0]-15, x[-1]+15)

    fig.subplots_adjust(
        left=0.12,
        right=0.96,
        bottom=0.08,
        top=0.97,
        hspace=0.05,
    )
    # save_figure(fig, output_path, show=False, tight=False)
    # plt.close(fig)


# =========================
# 加速度三轴图
# =========================
plot_3axis_vertical(
    x=x,
    data=[accX, accY, accZ],
    labels=["accX(m/s²)", "accY(m/s²)", "accZ(m/s²)"],
    line_color = "#1f77b4",
    output_path=ACC_OUTPUT_PATH,
    figsize=FIG_SIZE,
)

# =========================
# 陀螺仪三轴图
# =========================
plot_3axis_vertical(
    x=x,
    data=[gyroX, gyroY, gyroZ],
    labels=["gyroX(rad/s)", "gyroY(rad/s)", "gyroZ(rad/s)"],
    line_color="orange",
    output_path=GYRO_OUTPUT_PATH,
    figsize=FIG_SIZE,
)

print("绘图完成：")
print(ACC_OUTPUT_PATH)
print(GYRO_OUTPUT_PATH)