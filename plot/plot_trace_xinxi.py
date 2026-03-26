import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import io
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from plot.utils.plot_style import setup_plot_equal_style, style_axis, save_figure

setup_plot_equal_style()

ROOT = Path(__file__).resolve().parents[1]
CSV_PATHS = [
    # ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_phone_xinxi" / "2100_xinxi_test1_loc_res_meanerr_0.6565.csv",
    # ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_phone_xinxi" / "2100_xinxi_test4_loc_res_meanerr_0.4528.csv",
    # ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_phone_xinxi" / "2100_xinxi_test3_loc_res_meanerr_0.5791.csv",
    # ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_phone_xinxi" / "2100_xinxi_test2_loc_res_meanerr_0.4804.csv",
    
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_phone_xinxi_2" / "0145_xinxi_test5_loc_res_meanerr_0.5761.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_phone_xinxi_2" / "0145_xinxi_test8_loc_res_meanerr_0.5443.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_phone_xinxi_2" / "0145_xinxi_test7_loc_res_meanerr_0.6125.csv",
    ROOT / "runs" / "loc_res" / "mag_imu_eqnio_different_phone_xinxi_2" / "0145_xinxi_test6_loc_res_meanerr_0.5309.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_phone_xinxi_2" / "1817_xinxi_test5_loc_res_meanerr_0.7809.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_phone_xinxi_2" / "2141_xinxi_test8_loc_res_meanerr_0.7536.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_phone_xinxi_2" / "2141_xinxi_test7_loc_res_meanerr_0.8321.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_phone_xinxi_2" / "2141_xinxi_test6_lgitoc_res_meanerr_0.7204.csv",
]

LABELS = ["Huawei P60", "MEIZU 20", "OPPO Find X", "Xiaomi 14"]
COLORS = ["#0000FF", "#FF0000", "#9400D3","#F3A332"]
OUTPUT_PATH = ROOT / "figures" / "loc_traj_different_phone_xinxi_2.png"

def load_result_detail_csv(path):
    """
    读取两段式结果 csv，只保留明细部分，并强制转成数值列
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("pred_x,pred_y,true_x,true_y"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"在文件中未找到结果明细表头: {path}")

    content = "".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(content))

    numeric_cols = ["pred_x", "pred_y", "true_x", "true_y", "euclidean_error"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["pred_x", "pred_y", "true_x", "true_y"]).reset_index(drop=True)
    return df

def load_multi_trajectories(csv_paths):
    """
    读取多条轨迹：
    返回：
        true_x, true_y: (N,) 真值轨迹
        traj_list: 每个元素是 dict，包含 pred_x, pred_y, error
    """
    true_x, true_y = None, None
    traj_list = []

    for path in csv_paths:
        df = load_result_detail_csv(path)

        if len(df) == 0:
            raise ValueError(f"文件没有有效轨迹数据: {path}")

        if true_x is None:
            true_x = df["true_x"].to_numpy(dtype=float)
            true_y = df["true_y"].to_numpy(dtype=float)

        traj_list.append(
            dict(
                pred_x=df["pred_x"].to_numpy(dtype=float),
                pred_y=df["pred_y"].to_numpy(dtype=float),
                error=df["euclidean_error"].to_numpy(dtype=float)
                if "euclidean_error" in df.columns
                else None,
            )
        )

    return true_x, true_y, traj_list


# def plot_multi_trajectories(csv_paths, labels, output_path, title="Localization Trajectories"):
#     assert len(csv_paths) == len(labels), "CSV_PATHS 和 LABELS 数量必须一致"

#     true_x, true_y, traj_list = load_multi_trajectories(csv_paths)

#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     fig, ax = plt.subplots(figsize=(6, 6))
#     # assert true_x
#     # assert true_y
#     # 画真值轨迹（黑色实线，圆点）
#     ax.plot(true_x, true_y, linestyle="-", color="k", label="Ground Truth") # type: ignore

#     # 画多条预测轨迹
#     for traj, label in zip(traj_list, labels):
#         ax.plot(
#             traj["pred_x"],
#             traj["pred_y"],
#             linestyle="--",
#             # marker="x",
#             label=label,
#         )

#         # 如果想画真值到预测点的误差连线，可取消注释：
#         # for px, py, tx, ty in zip(traj["pred_x"], traj["pred_y"], true_x, true_y):
#         #     ax.plot([tx, px], [ty, py], linestyle=":", linewidth=0.8, alpha=0.4)

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True)
#     ax.axis("equal")  # 保持坐标比例一致

#     fig.tight_layout()
#     fig.savefig(output_path, dpi=300)
#     plt.close(fig)

def plot_multi_trajectories(
    csv_paths,
    labels,
    output_path,
):
    assert len(csv_paths) == len(labels), "CSV_PATHS 和 LABELS 数量必须一致"

    true_x, true_y, traj_list = load_multi_trajectories(csv_paths)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # fig, ax = plt.subplots(figsize=(6.3, 5.8))
    fig, ax = plt.subplots(figsize=(5.2, 6))
    ax.scatter(
        true_x[0], true_y[0],  # type: ignore
        s=80,                # 尺寸大一点，醒目
        marker="^",          # 向上三角形
        color="#00FF00",     # 鲜艳的绿色起点
        edgecolors="black",  # 黑色边框增加精致感
        linewidths=1,
        zorder=10,           # 确保在所有轨迹之上
        label="路径起点"
    )

    # ===== 真值轨迹：粗黑线 =====
    ax.plot(
        true_x,  # type: ignore
        true_y,  # type: ignore
        linestyle="-",
        color="black",
        linewidth=1,
        label="真实轨迹",
        zorder=2,
    ) 

    # ===== 多条预测轨迹 =====
    error_threshold = 3.0

    for idx, (traj, label) in enumerate(zip(traj_list, labels)):

        px = traj["pred_x"]
        py = traj["pred_y"]
        err = traj["error"]

        if err is None:
            # 没有误差列就直接画
            ax.plot(px, py, color=COLORS[idx], linewidth=1.2, label=label)
            continue

        px = pd.Series(px)
        py = pd.Series(py)
        err = pd.Series(err)

        # 正常点
        normal_mask = err <= error_threshold

        # 离群点
        outlier_mask = err > error_threshold

        # 正常轨迹（连续线）
        ax.plot(
            px[normal_mask],
            py[normal_mask],
            color=COLORS[idx],
            linewidth=1.2,
            alpha=0.8,
            label=label,
            zorder=3,
        )

        # # 离群点（散点）
        # ax.scatter(
        #     px[outlier_mask],
        #     py[outlier_mask],
        #     color=COLORS[idx],
        #     marker="o",
        #     s=10,
        #     alpha=0.9,
        #     zorder=4,
        #     label=f"{label} outlier" if idx == 0 else None
        # )

        # 离群点 (Outliers)
        if outlier_mask.any():
            ax.scatter(
                px[outlier_mask], py[outlier_mask],
                color=COLORS[idx], marker="o", s=12,
                edgecolors="white",
                alpha=0.9, zorder=4,
            )
    
    # ===== 4. 创建统一的离群点代理图例 =====
    # 创建一个灰色的圆点作为离群点的通用标识
    outlier_proxy = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                  markersize=4, markeredgecolor='white', label='离群点 (Outliers)')

    # ===== 5. 图例手动排序与整合 =====
    handles, labels_all = ax.get_legend_handles_labels()
    # 将代理图例加入列表末尾
    handles.append(outlier_proxy)
    labels_all.append('离群点')

    # ===== 轴标签 & 标题 =====
    ax.set_xlabel("x轴 (m)")
    ax.set_ylabel("y轴 (m)")
    ax.set_xlim(-3, 42.5)
    ax.set_ylim(-2, 49)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis="both", which="major", length=2, direction="in", top=True, right=True)

    # 应用美化后的图例
    ax.legend(
        handles=handles, labels=labels_all,
        loc="center",
        frameon=False,
        fontsize=9,
        ncol=1 # 如果图例太多，可以考虑设为 2
    )
    
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.05)
    ax.grid(False)

    # # ===== 8. 核心优化：手动 Hard-coding 留白 (替代 tight_layout) =====
    # # 只要两张图的这些参数完全一致，绘图区的大小就绝对统一
    # # 请根据需要微调这些百分比值
    # plt.gcf().subplots_adjust(left=0.01,   # 左边距 15%
    #                             bottom=0.08, # 底边距 15%
    #                             right=0.99,  # 右边距到 95%
    #                             top=0.92)   # 顶边距到 95%


    plt.tight_layout()
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    plot_multi_trajectories(CSV_PATHS, LABELS, OUTPUT_PATH)
