import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

from plot.utils.plot_style import setup_plot_equal_style, save_figure
from datasets.utils import gravity_align_per_window

setup_plot_equal_style()

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "gravity_align" / "dataset_2026-03-31_15-33-19-081.csv"

ACC_BEFORE_OUTPUT_PATH = ROOT / "figures" / "acc_before_alignment_free.png"
ACC_AFTER_OUTPUT_PATH = ROOT / "figures" / "acc_after_alignment_free.png"

GRAV_BEFORE_OUTPUT_PATH = ROOT / "figures" / "gravity_before_alignment_free.png"
GRAV_AFTER_OUTPUT_PATH = ROOT / "figures" / "gravity_after_alignment_free.png"

# =========================
# 参数
# =========================
WINDOW_START = 300
SEQ_LEN = 128
FIG_SIZE = (6.0, 5.0)

ACC_COLS = ["accX", "accY", "accZ"]
GYRO_COLS = ["gyroX", "gyroY", "gyroZ"]
GRAV_COLS = ["gravityX", "gravityY", "gravityZ"]
MAG_COLS = ["magX", "magY", "magZ"]   # 如果列名不同，请改成你自己的

# =========================
# 读取原始数据
# =========================
df = pd.read_csv(DATA_PATH)

need_cols = ACC_COLS + GYRO_COLS + GRAV_COLS + MAG_COLS
for col in need_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=need_cols).reset_index(drop=True)

if WINDOW_START + SEQ_LEN > len(df):
    raise ValueError(
        f"WINDOW_START + SEQ_LEN = {WINDOW_START + SEQ_LEN} 超出数据长度 {len(df)}"
    )

# =========================
# 模拟滑窗：从原始序列中取一个窗口
# =========================
window_df = df.iloc[WINDOW_START: WINDOW_START + SEQ_LEN].copy().reset_index(drop=True)

x = np.arange(len(window_df))

acc_win = window_df[ACC_COLS].to_numpy(dtype=np.float32)
gyro_win = window_df[GYRO_COLS].to_numpy(dtype=np.float32)
grav_win = window_df[GRAV_COLS].to_numpy(dtype=np.float32)
mag_win = window_df[MAG_COLS].to_numpy(dtype=np.float32)

# =========================
# 直接复用原始重力对齐函数
# =========================
mag_aligned, acc_aligned, gyro_aligned, grav_aligned = gravity_align_per_window(
    mag_win=mag_win,
    acc_win=acc_win,
    gyro_win=gyro_win,
    grav_win=grav_win,
    use_linear_acc=False,
)

# =========================
# 绘图函数：单张三轴图
# =========================
def plot_3axis(
    x,
    data,
    labels,
    color,
    output_path,
    figsize=(8.5, 6.0),
):
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for i in range(3):
        ax = axes[i]
        y = data[i]

        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.5,
        )

        ax.set_ylabel(labels[i])
        ax.grid(False)

        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            length=4,
            width=1.0,
        )

        mean_val = np.mean(y)

        max_dev = np.max(np.abs(y - mean_val))
        margin = max_dev * 1.2   # 留一点空间（1.1~1.3都可以）

        ax.set_ylim(mean_val - margin, mean_val + margin)

        max_abs_val = np.max(np.abs(y))
        if max_abs_val < 0.05:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    axes[-1].set_xlabel("样本序号")
    axes[-1].set_xlim(x[0]-3, x[-1]+3)

    fig.subplots_adjust(
        left=0.12,
        right=0.96,
        bottom=0.08,
        top=0.92,
        hspace=0.05,
    )

    save_figure(fig, output_path, show=False, tight=False)
    plt.close(fig)

# =========================
# 绘制加速度：对齐前
# =========================
plot_3axis(
    x=x,
    data=acc_win,
    labels=["accX(m/s²)", "accY(m/s²)", "accZ(m/s²)"],
    color="#1f77b4",
    output_path=ACC_BEFORE_OUTPUT_PATH,
    figsize=FIG_SIZE,
)

# =========================
# 绘制加速度：对齐后
# =========================
plot_3axis(
    x=x,
    data=acc_aligned,
    labels=["accX(m/s²)", "accY(m/s²)", "accZ(m/s²)"],
    color="orange",
    output_path=ACC_AFTER_OUTPUT_PATH,
    figsize=FIG_SIZE,
)

# # =========================
# # 绘制重力向量：对齐前
# # 如果你暂时不想画重力图，可以把下面两段注释掉
# # =========================
# plot_3axis(
#     x=x,
#     data=grav_win,
#     labels=["gravityX(m/s²)", "gravityY(m/s²)", "gravityZ(m/s²)"],
#     color="#fdd0a2",
#     title="Gravity before alignment",
#     output_path=GRAV_BEFORE_OUTPUT_PATH,
#     figsize=FIG_SIZE,
# )

# # =========================
# # 绘制重力向量：对齐后
# # =========================
# plot_3axis(
#     x=x,
#     data=grav_aligned,
#     labels=["gravityX(m/s²)", "gravityY(m/s²)", "gravityZ(m/s²)"],
#     color="#ff7f0e",
#     title="Gravity after alignment",
#     output_path=GRAV_AFTER_OUTPUT_PATH,
#     figsize=FIG_SIZE,
# )

print("绘图完成：")
print(ACC_BEFORE_OUTPUT_PATH)
print(ACC_AFTER_OUTPUT_PATH)
# print(GRAV_BEFORE_OUTPUT_PATH)
# print(GRAV_AFTER_OUTPUT_PATH)