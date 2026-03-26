import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import matplotlib.patheffects as pe
from plot.utils.plot_style import setup_plot_equivalent_style, style_axis, save_figure

setup_plot_equivalent_style()

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "figures" / "can_loss_bar.png"


# ===== 数据 =====
bar_width = 0.06
groups = ['信息学馆-路径1', '信息学馆-路径2', '文管学馆-路径1']
values = [
    [0.50, 0.51, 0.72],
    [1.05, 0.98, 1.22],
    [1.53, 1.95, 2.10],
]
labels = ['Full', 'No-Aug', 'No-Cons']

# 每组中心（按组数量自动生成，组间距略缩小）
x_group = np.arange(len(groups)) * 0.3
category_num = len(values)
offsets = np.linspace(
    - (category_num - 1) * bar_width / 2,
    + (category_num - 1) * bar_width / 2,
    category_num
)

# colors = [
#   "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
# ]

colors = [
  "#EAEAEA",  # light gray
  "#C8DFFF",  # soft sky blue
  "#CCCCCC",  # medium gray
  "#7EA4D3",  # steel blue
]

# colors = [
# #   "#EAEAEA",  # light gray
#     "#FFC300",
#   "#FDE4AD",  # soft sky blue
#   "#D1EDF3",  # medium gray
#   "#FCD7D4",  # steel blue
# ]

plt.figure(figsize=(10, 8))

for i in range(category_num):
    bar_positions = x_group + offsets[i]
    
    # 绘制柱状图
    plt.bar(
        bar_positions,
        values[i],
        width=bar_width,
        color=colors[i],
        edgecolor="black",
        linewidth=2.0,
        label=labels[i],
    )

    # ===== 添加数值标记 =====
    for j, x_pos in enumerate(bar_positions):
        plt.text(
            x_pos, 
            values[i][j] + 0.005,       # 文字放在柱子稍上方
            f"{values[i][j]:.2f}",
            ha='center',
            va='bottom',
            fontsize=15
        )

# 坐标轴
plt.xticks(x_group, groups)
plt.ylabel("平均定位误差(m)")
plt.ylim(0, 3)

plt.legend(title="", loc='best')
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=600)
plt.close()

print(f"Saved CDF plot to {OUTPUT_PATH.resolve()}")
