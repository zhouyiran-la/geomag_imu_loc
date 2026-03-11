from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_plot_equal_style():
    """
    全局 Matplotlib 样式设置。
    在每个绘图脚本开头调用一次即可。
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
       "Source Han Sans SC"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    matplotlib.rcParams.update({
        
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.4,
        "axes.edgecolor": "0.25",
        "axes.linewidth": 1.5,
        "savefig.dpi": 600,
        "figure.dpi": 150,
    })


def setup_plot_season_trend_style():
    """
    全局 Matplotlib 样式设置。
    在每个绘图脚本开头调用一次即可。
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt_rc = matplotlib.rcParams
    plt_rc["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt_rc["axes.unicode_minus"] = False
    plt_rc.update({
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 13,
        "lines.linewidth": 1.2,
        "grid.alpha": 0.4,
        "axes.edgecolor": "0.25",
        "axes.linewidth": 1.5,
        "savefig.dpi": 600,
        "figure.dpi": 150,
    })


def style_axis(ax, title: str, xlabel: str = "Time Step", ylabel: str = "Value"):
    """
    统一坐标轴样式。
    """
    if title != None:
        ax.set_title(title, pad=8)
    if xlabel != None:
        ax.set_xlabel(xlabel)
    if ylabel != None:
        ax.set_ylabel(ylabel)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("0.25")


def save_figure(fig, save_path: Path, show: bool = False, tight: bool = True):
    """
    统一保存图片。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if tight:
        fig.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"图像已保存到: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)