import csv
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from matplotlib import font_manager
import matplotlib
from plot.utils.plot_style import setup_plot_equivalent_style, style_axis, save_figure

setup_plot_equivalent_style()

# matplotlib.rcParams["axes.unicode_minus"] = False

# matplotlib.use("Agg")
# plt.style.use("seaborn-v0_8-whitegrid")
# plt_rc = matplotlib.rcParams

# plt_rc["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
# plt_rc["axes.unicode_minus"] = False
# plt_rc.update({
#     "axes.labelsize": 15,
#     "axes.titlesize": 15,
#     "xtick.labelsize": 13,
#     "ytick.labelsize": 13,
#     "legend.fontsize": 13,
#     "lines.linewidth": 2.0,
#     "grid.alpha": 0.4,
#     "axes.edgecolor": "0.25",
#     "axes.linewidth": 1.5,
# })

ROOT = Path(__file__).resolve().parents[1]
CSV_PATHS = [

    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "2128_wenguan_test2_loc_res_meanerr_0.5608.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "1738_wenguan_test1_loc_res_can_mag_meanerr_0.9118.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "2047_wenguan_test2_loc_res_can_imu_meanerr_1.3095.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "1717_wenguan_test3_no_framenet_trans_loc_meanerr_1.2447.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "1732_wenguan_test2_loc_res_no_framenet_ga_meanerr_2.0377.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_equivalent_wenguan" / "1705_wenguan_test1_loc_res_no_framenet_meanerr_3.7131.csv",
    
]

# LABELS = ["Proposed", "Wang(2024)", "HLSTM(2022)", "MAIL(2020)", "RNN"]
LABELS = ["M1", "M2", "M3", "M4" , "M5", "M6"]
OUTPUT_PATH = ROOT / "figures" / "loc_cdf_equivalent_wenguan.png"
PLOT_TITLE = "Localization Error CDF"
X_MAX = None  # Set to a float to force xmax, or None to auto-scale.


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


def plot_cdf(errors: np.ndarray):
    sorted_errors = np.sort(errors)
    xs = np.concatenate(([0.0], sorted_errors))
    n = len(sorted_errors)
    probs = np.arange(n + 1, dtype=np.float32) / max(n, 1)
    return xs, probs


def main():
    if LABELS and len(LABELS) != len(CSV_PATHS):
        raise ValueError("Number of labels must match number of CSV files.")
    labels = LABELS or [p.stem for p in CSV_PATHS]

    pairs: list[tuple[Path, str]] = []
    missing = []
    for path, label in zip(CSV_PATHS, labels):
        if path.exists():
            pairs.append((path, label))
        else:
            missing.append(str(path))

    if not pairs:
        raise FileNotFoundError("No CSV files found. Please check CSV_PATHS.")
    if missing:
        print(f"Skipping missing files: {', '.join(missing)}")

    plt.figure(figsize=(10, 8))
    curves = []
    max_x = 0.0
    for path, label in pairs:
        errors = read_errors(path)
        xs, ys = plot_cdf(errors)
        curves.append((xs, ys, label))
        if xs.size:
            max_x = max(max_x, float(xs.max()))

    for xs, ys, label in curves:
        plt.plot(xs, ys, label=label)

    plt.xlabel("定位误差（m）")
    plt.ylabel("概率")
    # plt.title(PLOT_TITLE)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="lower right")
    
    plt.xticks(np.linspace(0, 20, num=5))
    plt.yticks(np.linspace(0.0, 1.0, num=6))
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.6)
    plt.xlim(-0.5, 19)
    plt.ylim(0, 1.0)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=600)
    plt.close()
    print(f"Saved CDF plot to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
