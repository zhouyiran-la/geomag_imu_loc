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

    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "2128_wenguan_test2_loc_res_meanerr_0.5608.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "1738_wenguan_test1_loc_res_can_mag_meanerr_0.9118.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "2047_wenguan_test2_loc_res_can_imu_meanerr_1.3095.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "1717_wenguan_test3_no_framenet_trans_loc_meanerr_1.2447.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "1732_wenguan_test2_loc_res_no_framenet_ga_meanerr_2.0377.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "1705_wenguan_test1_loc_res_no_framenet_meanerr_3.7131.csv",
    
]

LABELS = ["M1", "M2", "M3", "M4" , "M5", "M6"]

OUTPUT_PATH = ROOT / "figures" / "loc_error_boxplot_equivalent_wenguan.png"
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

if __name__ == "__main__":
    main()
