"""One type size, line weight, and palette for the paper's data figures (Scientific Reports: 89 or 183 mm wide,
text 5-7 pt at final size, vector PDF).

Colour carries identity in two separate ways. Methods: the belt and its reference rate in ink, PARH-OSSM in blue,
the single-observation OSSM-KF in vermilion. Measurement regions: the chest strip in teal-green and the torso box
in purple, which colour only an observation's update share and the region marks of Figure 1, never a method.
The observation traces themselves are one dark grey, and guides and the constant predictor are lighter grey.

Display of a trace (both figures): centre on the median, scale by the median absolute deviation, and compress as
3 tanh(z / 3) so that a motion burst stays inside its row; the samples more than BEYOND deviations from the
median, where the compression flattens the trace, are drawn in a lighter tone of the trace's own colour.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

MM = 1 / 25.4
INK, PARH, OSSM, NEUTRAL, GUIDE = "#222222", "#1f5aa6", "#c8553d", "#8c8c8c", "#9a9a9a"
CHEST, TORSO = "#16866a", "#7d3c98"
OBS = "#505050"
BEYOND = 4

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5, "axes.linewidth": 0.5,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4, "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": True, "pdf.fonttype": 42,
    "lines.solid_capstyle": "butt", "legend.frameon": False,
})


def panel_letter(ax, letter, x, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def lighter(color, amount=0.55):
    """The colour mixed with white; the tone of compressed samples."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


def tint(color, amount=0.72):
    """A light tint of a region colour, for the area of an update share."""
    return lighter(color, amount)


def median_z(x):
    """x centred on its median and scaled by its median absolute deviation."""
    x = np.asarray(x, float)
    spread = 1.4826 * np.median(np.abs(x - np.median(x)))
    return (x - np.median(x)) / spread if spread > 0 else np.zeros_like(x)


def draw_trace(ax, time, z, color, offset=0.0, gain=1.0, lw=0.5, **kwargs):
    """Draw the scaled trace z as 3 tanh(z / 3); samples beyond BEYOND deviations get the lighter tone.
    Returns the boolean mask of those samples."""
    shown = 3 * np.tanh(z / 3) * gain + offset
    beyond = np.abs(z) > BEYOND
    joined = beyond | np.r_[beyond[1:], False] | np.r_[False, beyond[:-1]]
    ax.plot(time, np.where(beyond, np.nan, shown), color=color, lw=lw, **kwargs)
    ax.plot(time, np.where(joined, shown, np.nan), color=lighter(color), lw=lw)
    return beyond


def draw_share(ax, time, share, color):
    """The update share of one observation: an area under a thin zero line, in a light tint of its region colour."""
    ax.fill_between(time, 0, share, color=tint(color), lw=0)
    ax.axhline(0, color=NEUTRAL, lw=0.3)
    ax.set_ylim(0, 1)
