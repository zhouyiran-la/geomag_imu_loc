import csv
import numpy as np
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse

from plot.utils.plot_style import setup_plot_equivalent_style

setup_plot_equivalent_style()

# ================== 路径配置 ==================
ROOT = Path(__file__).resolve().parents[1]

CSV_PATHS = [
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_posture" /"1653_xinxi_test1_loc_res_meanerr_0.4705.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_posture" / "1432_xinxi_test4_loc_res_meanerr_0.5534.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_posture" /"1432_xinxi_test3_loc_res_meanerr_0.6432.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_posture" / "1653_xinxi_test3_loc_res_meanerr_0.6513.csv",
]

POSE_LABELS = ["水平", "竖直", "口袋", "任意"]

OUTPUT_PATH = ROOT / "figures" / "pose_xy_error_heatmap_xinxi.png"


# ================== 读取xy误差 ==================
def read_xy_error(csv_path: Path):
    dx_list: List[float] = []
    dy_list: List[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        header_line = None
        while True:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            if "pred_x" in s:
                header_line = s
                break

        if header_line is None:
            raise ValueError(f"Header not found in {csv_path}")

        headers = [h.strip() for h in header_line.split(",")]

        reader = csv.DictReader(f, fieldnames=headers)
        for row in reader:
            try:
                px = float(row["pred_x"])
                py = float(row["pred_y"])
                gx = float(row["true_x"])
                gy = float(row["true_y"])

                dx_list.append(px - gx)
                dy_list.append(py - gy)

            except:
                continue

    return np.array(dx_list), np.array(dy_list)

def draw_confidence_ellipse(x, y, ax, n_std=2.4477):
    """
    绘制二维95%置信椭圆
    n_std=2.4477 对应95%（二维高斯）
    """
    if x.size < 2:
        return

    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 排序（大→小）
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # 椭圆方向角
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    # 长轴、短轴
    width, height = 2 * n_std * np.sqrt(eigvals)

    ellipse = Ellipse(
        (mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor='#D4B0E1',
        alpha=0.35
    )

    ax.add_patch(ellipse)
# ================== 主函数 ==================
def main():

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.flatten()

    # -------- 统一坐标范围 --------
    all_dx = []
    all_dy = []

    data_list = []

    for p in CSV_PATHS:
        dx, dy = read_xy_error(p)
        data_list.append((dx, dy))
        all_dx.extend(dx)
        all_dy.extend(dy)

    lim = max(np.max(np.abs(all_dx)), np.max(np.abs(all_dy))) * 1.1

    # -------- 绘图 --------
    for ax, (dx, dy), label in zip(axes, data_list, POSE_LABELS):

        # KDE密度
        xy = np.vstack([dx, dy])
        z = gaussian_kde(xy)(xy)

        # 按密度排序（避免遮挡）
        idx = z.argsort()
        dx, dy, z = dx[idx], dy[idx], z[idx]

        sc = ax.scatter(
            dx,
            dy,
            c=z,
            s=10,
            cmap="jet",
            edgecolors="none"
        )
        draw_confidence_ellipse(dx, dy, ax)

        # 中心线（关键）
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.axvline(0, linestyle="--", linewidth=1)

        ax.set_title(label)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        ax.grid(False)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(direction="in")

    # 坐标标签
    axes[2].set_xlabel("x方向误差（m）")
    axes[3].set_xlabel("x方向误差（m）")
    axes[0].set_ylabel("y方向误差（m）")
    axes[2].set_ylabel("y方向误差（m）")
    
   
    cbar = fig.colorbar(
        sc, # type: ignore
        ax=axes,
        fraction=0.03,
        pad=0.02
    )
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=600)
    plt.close()

    print(f"Saved XY error heatmap to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()