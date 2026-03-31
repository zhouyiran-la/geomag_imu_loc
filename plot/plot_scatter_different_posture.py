import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from plot.utils.plot_style import setup_plot_equivalent_style

setup_plot_equivalent_style()

# ================== 路径配置 ==================
ROOT = Path(__file__).resolve().parents[1]

CSV_PATHS = [
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_posture" /"1653_xinxi_test1_loc_res_meanerr_0.4705.csv",
    ROOT / "runs" / "loc_res" /"mag_imu_eqnio_different_posture" / "1653_xinxi_test4_loc_res_meanerr_0.4872.csv",
    ROOT / "runs" / "loc_res" /"mag_imu_eqnio_different_posture" / "0145_xinxi_test3_loc_res_meanerr_0.6463.csv",
    ROOT / "runs" / "loc_res" /"mag_imu_eqnio_different_posture" / "1653_xinxi_test3_loc_res_meanerr_0.6513.csv",
]

POSE_LABELS = ["水平", "竖直", "姿态3", "姿态4"]

OUTPUT_PATH = ROOT / "figures" / "xinxi_path1_pose_error_scatter.png"
Y_MAX = None   # 可手动设为如 8.0


# ================== 读取误差 ==================
def read_errors(csv_path: Path) -> np.ndarray:
    errors: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        header_line = None
        while True:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            if "euclidean_error" in s.split(","):
                header_line = s
                break

        if header_line is None:
            raise ValueError(f"Data header with 'euclidean_error' not found in {csv_path}")

        headers = [h.strip() for h in header_line.split(",")]
        if "euclidean_error" not in headers:
            raise ValueError(f"'euclidean_error' column not found in {csv_path}")

        reader = csv.DictReader(f, fieldnames=headers)
        for row in reader:
            try:
                v = row.get("euclidean_error", "")
                if v is None or v == "":
                    continue
                errors.append(float(v))
            except (TypeError, ValueError):
                continue

    if not errors:
        raise ValueError(f"No valid errors loaded from {csv_path}")
    return np.asarray(errors, dtype=np.float32)


# ================== 主函数 ==================
def main():
    if len(CSV_PATHS) != 4 or len(POSE_LABELS) != 4:
        raise ValueError("需要提供4个CSV路径和4个姿态标签")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=True)
    axes = axes.flatten()

    global_ymax = 0.0
    all_errors = []

    for p in CSV_PATHS:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        errs = read_errors(p)
        all_errors.append(errs)
        global_ymax = max(global_ymax, float(np.max(errs)))

    if Y_MAX is None:
        y_upper = global_ymax * 1.1
    else:
        y_upper = Y_MAX

    point_color = "#4C72B0"
    mean_color = "#D62728"
    q95_color = "#2CA02C"

    for ax, errs, label in zip(axes, all_errors, POSE_LABELS):
        x = np.arange(1, len(errs) + 1)

        mean_err = float(np.mean(errs))
        median_err = float(np.median(errs))
        q95_err = float(np.percentile(errs, 95))

        ax.scatter(
            x,
            errs,
            s=16,
            alpha=0.75,
            color=point_color,
            edgecolors="none",
        )

        ax.axhline(mean_err, color=mean_color, linestyle="--", linewidth=1.4, label="均值")
        ax.axhline(q95_err, color=q95_color, linestyle="-.", linewidth=1.4, label="95%分位")

        ax.text(
            0.97, 0.95,
            f"Mean={mean_err:.3f} m\nMedian={median_err:.3f} m\n95%={q95_err:.3f} m",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )

        ax.set_title(label)
        ax.set_ylim(0, y_upper)

        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="major", direction="in", length=6, width=0.8)
        ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.6)

    axes[0].set_ylabel("定位误差（m）")
    axes[2].set_ylabel("定位误差（m）")
    axes[2].set_xlabel("轨迹点序号")
    axes[3].set_xlabel("轨迹点序号")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # type: ignore
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=600)
    plt.close()

    print(f"Saved pose error scatter figure to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()