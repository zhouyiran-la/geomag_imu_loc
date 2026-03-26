import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from plot.utils.plot_style import setup_plot_equivalent_style, style_axis, save_figure
from matplotlib.ticker import AutoMinorLocator

setup_plot_equivalent_style()

# ================== 路径配置 ==================
ROOT = Path(__file__).resolve().parents[1]

CSV_PATHS = [
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_xinxi" / "0145_xinxi_test2_loc_res_meanerr_0.5149.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_xinxi" / "1233_xinxi_test4_loc_res_can_mag_meanerr_0.7956.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_xinxi" / "2100_xinxi_test2_can_imu_loc_meanerr_1.4600.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_xinxi" / "2229_xinxi_test1_loc_res_no_framenet_trans_meanerr_1.3985.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_xinxi" / "2238_xinxi_test1_loc_res_no_framenet_ga_meanerr_1.6783.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_xinxi" / "2220_xinxi_test1_loc_res_no_framenet_meanerr_3.3611.csv",
    
]

LABELS = ["M1", "M2", "M3", "M4" , "M5", "M6"]

OUTPUT_PATH = ROOT / "figures" / "loc_error_boxplot_equivalent_xinxi.png"
Y_MAX = None  # 可手动设上限，如 6.0

# ================== 读取误差 ==================
def read_errors(csv_path: Path) -> np.ndarray:
    errors: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        # 先逐行读，直到找到真正数据表头（包含 euclidean_error）
        header_line = None
        while True:
            line = f.readline()
            if not line:  # EOF
                break
            s = line.strip()
            if not s:
                continue
            # 找到第二段的表头行
            if "euclidean_error" in s.split(","):
                header_line = s
                break

        if header_line is None:
            raise ValueError(f"Data header with 'euclidean_error' not found in {csv_path}")

        headers = [h.strip() for h in header_line.split(",")]
        if "euclidean_error" not in headers:
            raise ValueError(f"'euclidean_error' column not found in {csv_path}")

        # 从当前文件指针位置继续读数据行
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
    if len(LABELS) != len(CSV_PATHS):
        raise ValueError("LABELS length must match CSV_PATHS length")

    all_errors: List[np.ndarray] = []
    means: List[float] = []
    used_labels: List[str] = []
    y_max_seen = 0.0

    for p, lab in zip(CSV_PATHS, LABELS):
        if not p.exists():
            print(f"Skip missing file: {p}")
            continue
        errs = read_errors(p)
        all_errors.append(errs)
        means.append(float(np.mean(errs)))
        used_labels.append(lab)
        y_max_seen = max(y_max_seen, float(np.max(errs)))

    if not all_errors:
        raise RuntimeError("No valid CSV files found.")

    positions = np.arange(1, len(all_errors) + 1)

    flier_style = dict(
        marker='o', 
        markerfacecolor='black', 
        markeredgecolor='none', 
        markersize=4, 
    )
    plt.figure(figsize=(10, 8))
    
    # -------- 箱线图 --------
    plt.boxplot(
        all_errors,
        positions=positions,
        widths=0.55,
        whis=1.5,
        showfliers=True,
        patch_artist=True,
        flierprops=flier_style,
        boxprops=dict(facecolor='#AED6F1', linewidth=1.5),
        medianprops=dict(color="#B5711E", linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )

    # -------- 均值点 --------
    plt.scatter(
        positions,
        means,
        marker="o",
        s=40,
        zorder=3,
        label="均值",
    )

    plt.ylabel("定位误差（m）")
    plt.xticks(positions, used_labels, rotation=15)
    plt.grid(False)

    ax = plt.gca()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.6)

    
    plt.ylim(0, 19)

    plt.legend(loc="upper right")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=600)
    plt.close()

    print(f"Saved boxplot with mean to {OUTPUT_PATH.resolve()}")


# # ================== 改进的绘图逻辑 ==================
# def main():
#     if len(LABELS) != len(CSV_PATHS):
#         raise ValueError("LABELS length must match CSV_PATHS length")

#     all_errors: List[np.ndarray] = []
#     means: List[float] = []
#     used_labels: List[str] = []
    
#     for p, lab in zip(CSV_PATHS, LABELS):
#         if not p.exists():
#             print(f"Skip missing file: {p}")
#             continue
#         errs = read_errors(p)
#         all_errors.append(errs)
#         means.append(float(np.mean(errs)))
#         used_labels.append(lab)

#     if not all_errors:
#         raise RuntimeError("No valid CSV files found.")

#     positions = np.arange(1, len(all_errors) + 1)
    
#     # --- 科研配色配置 ---
#     # 使用学术常用的 Tab10 或 Set1 配色
#     colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))
#     box_color = '#2F4F4F'      # 深灰色线条，比纯黑更有质感
#     median_color = '#D62728'   # 鲜明的红色中位数线
#     mean_point_color = '#1f77b4' # 经典的学术蓝
#     patch_fill_color = '#E6E6E6' # 浅灰色填充，增加箱体存在感

#     fig, ax = plt.subplots(figsize=(10, 8))

#     # -------- 改进的箱线图 --------
#     bplot = ax.boxplot(
#         all_errors,
#         positions=positions,
#         widths=0.5,
#         whis=1.5,
#         showfliers=True,
#         patch_artist=True,  # 开启填充
#         # 异常值样式：更小、更淡的空心圆，避免抢戏
#         flierprops={'marker': 'o', 'markersize': 4, 'markeredgecolor': '#999999', 'alpha': 0.5},
#         # 中位数线样式
#         medianprops={'color': median_color, 'linewidth': 2},
#         # 箱体边框样式
#         boxprops={'color': box_color, 'linewidth': 1.2},
#         # 须线样式
#         whiskerprops={'color': box_color, 'linewidth': 1.2, 'linestyle': '--'},
#         capprops={'color': box_color, 'linewidth': 1.2}
#     )

#     # 填充箱体颜色（带一点透明度）
#     for patch in bplot['boxes']:
#         patch.set_facecolor(patch_fill_color)
#         patch.set_alpha(0.7)

#     # -------- 均值点（增加描边提升对比度） --------
#     ax.scatter(
#         positions,
#         means,
#         marker="D",         # 改为菱形，与圆形的异常值区分
#         s=50,
#         color=mean_point_color,
#         edgecolors='white', # 白色描边使点更突出
#         linewidths=1,
#         zorder=4,
#         label="Mean Error",
#     )

#     # -------- 坐标轴与网格优化 --------
#     ax.set_ylabel("定位误差 (m)", fontweight='bold')
#     ax.set_xticks(positions)
#     ax.set_xticklabels(used_labels, rotation=0) # 若标签短，0度更易读
    
#     # 只开启Y轴主要网格，减少视觉干扰
#     ax.grid(axis='y', linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)

#     # 细化刻度
#     ax.yaxis.set_minor_locator(AutoMinorLocator(2))
#     ax.tick_params(axis="both", which="major", direction="in", length=6)
#     ax.tick_params(axis="both", which="minor", direction="in", length=3)

#     # 设置Y轴范围
#     ax.set_ylim(0, 22.9)

#     # 图例美化
#     ax.legend(loc="upper right")

#     # 保存
#     OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(OUTPUT_PATH, dpi=600, bbox_inches='tight')
#     plt.close()

#     print(f"Saved optimized boxplot to {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
