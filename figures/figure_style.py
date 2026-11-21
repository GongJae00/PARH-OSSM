"""One type size, line weight, and palette for the paper's data figures (Scientific Reports: 89 or 183 mm wide,
text 5-7 pt at final size, vector PDF).

Colour carries identity only where a figure compares methods: the belt and its reference rate in ink,
PARH-OSSM in blue, the single-observation OSSM-KF in vermilion. Everything else, such as observations, guides,
and the constant predictor, is grey.
"""
import matplotlib.pyplot as plt

MM = 1 / 25.4
INK, PARH, OSSM, NEUTRAL, GUIDE = "#222222", "#1f5aa6", "#c8553d", "#8c8c8c", "#9a9a9a"

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5, "axes.linewidth": 0.5,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4, "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": True, "pdf.fonttype": 42,
    "lines.solid_capstyle": "butt", "legend.frameon": False,
})


def panel_letter(ax, letter, x, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
