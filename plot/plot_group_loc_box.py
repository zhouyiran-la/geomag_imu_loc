import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator

from plot.utils.plot_style import setup_plot_equivalent_style

setup_plot_equivalent_style()

# ================== 路径配置 ==================
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "figures" / "loc_error_group_boxplot.png"
Y_MAX = 10
# ================== 姿态在外层 ==================
CSV_GROUPS = {
    "信息学馆-路径1": {
        "水平": ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_posture" /"1653_xinxi_test1_loc_res_meanerr_0.4705.csv",
        "竖直": ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_posture" / "1432_xinxi_test4_loc_res_meanerr_0.5534.csv",
        "口袋": ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_posture" /"1432_xinxi_test3_loc_res_meanerr_0.6432.csv",
        "任意": ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_posture" / "1653_xinxi_test3_loc_res_meanerr_0.6513.csv",
    },
    "信息学馆-路径2": {
        "水平": ROOT / "runs" / "loc_res" /"mag_imu_eqnio_different_posture" /"1653_xinxi_2_test6_loc_res_meanerr_0.3054.csv",
        "竖直": ROOT / "runs" / "loc_res" /"mag_imu_eqnio_different_posture" / "1653_xinxi_2_test8_loc_res_meanerr_0.4887.csv",
        "口袋": ROOT / "runs" / "loc_res" /"mag_imu_eqnio_different_posture" /"1653_xinxi_2_test5_loc_res_meanerr_0.4822.csv",
        "任意": ROOT / "runs" / "loc_res" /"mag_imu_eqnio_different_posture" /"1653_xinxi_2_test7_loc_res_meanerr_0.5720.csv",
    },
    "文管学馆-路径1": {
        "水平": ROOT / "runs" / "loc_res" /"mag_imu_eqnio_different_posture" /"2128_wenguan_test3_loc_meanerr_0.5906.csv",
        "竖直": ROOT / "runs" / "loc_res" /"mag_imu_eqnio_different_posture" / "2314_wenguan_test2_loc_res_meanerr_0.5483.csv",
        "口袋": ROOT / "runs" / "loc_res" /"mag_imu_eqnio_different_posture" /"2314_wenguan_test3_loc_res_meanerr_0.6874.csv",
        "任意": ROOT / "runs" / "loc_res" /"mag_imu_eqnio_different_posture" /"2202_wenguan_test3_loc_res_meanerr_0.7651.csv",
    },
}

POSE_ORDER = ["水平", "竖直", "口袋", "任意"]

POSE_COLORS = {
    "水平": "#4C72B0",
    "竖直": "#DD8452",
    "口袋": "#55A868",
    "任意": "#C44E52",
}


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


def plot_grouped_boxplot():

    scene_names = list(CSV_GROUPS.keys())
    n_scene = len(scene_names)
    n_pose = len(POSE_ORDER)

    fig, ax = plt.subplots(figsize=(8,7))

    group_gap = 1.8
    box_width = 0.28
    intra_gap = box_width * 0.85   # 控制紧凑度


    all_positions = []
    all_errors = []
    all_colors = []
    scene_centers = []

    for i, scene in enumerate(scene_names):
        base = i * group_gap
        current_positions = []
        offset = (n_pose - 1) * intra_gap / 2

        for j, pose in enumerate(POSE_ORDER):
            csv_path = CSV_GROUPS[scene].get(pose, None)

            if csv_path is None or not csv_path.exists():
                print(f"Skip: {scene}-{pose}")
                continue
            pos = base + j * intra_gap - offset
            errs = read_errors(csv_path)
            all_positions.append(pos)
            all_errors.append(errs)
            all_colors.append(POSE_COLORS[pose])
            current_positions.append(pos)
        scene_centers.append(np.mean(current_positions))
            

    flier_style = dict(
        marker='o', 
        markerfacecolor='black', 
        markeredgecolor='none', 
        markersize=4, 
    )

    # -------- 箱线图 --------
    bplot = ax.boxplot(
        all_errors,
        positions=all_positions,
        widths=box_width * 0.85,
        patch_artist=True,
        showfliers=True,
        flierprops=flier_style,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.5),
    )

    # 上色（按姿态）
    for patch, color in zip(bplot["boxes"], all_colors):
        patch.set_facecolor(color)

    # 横轴：场景
    ax.set_xticks(scene_centers)
    ax.set_xticklabels(scene_names)
    ax.set_ylim(0, 7.3)
    ax.set_ylabel("定位误差（m）")

    # 网格
    ax.grid(False)

    # 图例：姿态
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=POSE_COLORS[p], label=p)
        for p in POSE_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    # 美化
    ax.tick_params(direction="in")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=600)
    plt.close()

    print(f"Saved grouped boxplot to {OUTPUT_PATH}")



if __name__ == "__main__":
    plot_grouped_boxplot()