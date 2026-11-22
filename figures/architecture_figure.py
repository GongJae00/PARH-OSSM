"""Figure 1: the pipeline of PARH-OSSM from video to outputs, drawn at the text width of the article (175.9 mm) so
that LaTeX does not scale it.

Top row: (a) the video, (b) the regions placed on its first frame, (c) the nine observations of the extraction,
grouped by image measurement, with the three of the bank framed. Bottom
row: (d) the reliability-weighted observation law, (e) the oscillator state-space model and its smoothed state,
(f) the two outputs. The photographs are raster crops of the nine-observation version of the submitted figure, whose
photographs show an author, and illustrate the regions only. Each observation is drawn with its operator image from the submitted
figure (frame stacks of the flow and difference images, the correlation curve and its fitted interpolant for the P1D
routes, the torso box on the photograph). Every trace is the
stored output of one development trial of the frozen run (MAHNOB-HCI session 10) over 5-50 s: the observations
band-passed to 0.08-0.5 Hz and scaled by their median absolute deviation, the share of each observation in the
Kalman update, w_k = (1 / R_eff,k) / sum_j (1 / R_eff,j) from R_eff_t_channels, the RTS-smoothed state components
on the scale of the fundamental, the rate track at the window centres, and the smoothed waveform. A light guide
marks the transient, the time of the largest median over the observations of |z|.

usage: architecture_figure.py <nine-observation figure png> <observation pkl> <final pkl> <window rates csv> <out.pdf>
"""
import json
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from scipy.signal import butter, sosfiltfilt

matplotlib.use("pdf")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from figure_style import CHEST, GUIDE, INK, MM, NEUTRAL, OBS, PARH, TORSO, draw_trace, median_z, tint  # noqa: E402

plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "custom",
                     "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
                     "mathtext.bf": "Liberation Sans:bold"})

source_png, observation_pkl, final_pkl, rates_csv, out_pdf = sys.argv[1:6]
WIDTH, HEIGHT = 175.9, 84.0
SPAN_S, BAND_HZ = (5.0, 50.0), (0.08, 0.5)
FRAME, BLOCK = "#c8c8c8", "#4a4a4a"
# observation family of the run -> (name in the pickle of observations, label, region colour)
FAMILIES = {"of_farneback": ("of_farneback", "OF", CHEST),
            "profile1d_quadratic": ("profile1D quadratic", "P1D", CHEST),
            "torso_shift": ("torso_shift", "Torso displacement", TORSO)}
COMPONENTS = [("h1", "Fundamental", 1), ("h2", "Second harmonic", 1), ("baseline", "Baseline", 1),
              ("residual", "Residual (×10)", 10)]

# raster sources, in pixels of the nine-observation figure
figure = np.array(Image.open(source_png).convert("RGB"))
frame = figure[58:255, 74:348]
CHEST_BOX, TORSO_BOX = (66, 131, 210, 163), (52, 124, 219, 196)                       # x0, y0, x1, y1 of frame

observations = pickle.load(open(observation_pkl, "rb"))
fps = float(observations["fps"])
signals = {e["method"]: np.asarray(e["estimate"]["signal_hat"], float).reshape(-1) for e in observations["estimates"]}
estimate = pickle.load(open(final_pkl, "rb"))["estimates"][0]["estimate"]
families = json.loads(estimate["meta"])["observation_families"]
precision = 1.0 / np.asarray(estimate["diagnostics"]["R_eff_t_channels"], float)
update_share = precision / precision.sum(axis=0)
trial = final_pkl.rsplit("/", 1)[1][:-4]
windows = pd.read_csv(rates_csv).query("trial == @trial")


def band_passed(x):
    return sosfiltfilt(butter(2, BAND_HZ, btype="band", fs=fps, output="sos"), x)


def shown(x):
    """Time (s) and values of x inside the displayed span."""
    time = np.arange(x.size) / fps
    keep = (time >= SPAN_S[0]) & (time <= SPAN_S[1])
    return time[keep], x[keep]


scaled = {family: median_z(band_passed(signals[FAMILIES[family][0]])) for family in families}
n_common = min(z.size for z in scaled.values())
median_abs = np.median([np.abs(z[:n_common]) for z in scaled.values()], axis=0)
common_time = np.arange(n_common) / fps
in_span = (common_time >= SPAN_S[0]) & (common_time <= SPAN_S[1])
TRANSIENT_S = float(common_time[np.argmax(np.where(in_span, median_abs, -1))])

fig = plt.figure(figsize=(WIDTH * MM, HEIGHT * MM))
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
ax.axis("off")


def axes_at(x0, y0, x1, y1, xlim=SPAN_S, ylim=(-3.3, 3.3)):
    """An axes over the figure rectangle (x0, y0)-(x1, y1) in mm, without frame or ticks."""
    a = fig.add_axes((x0 / WIDTH, 1 - y1 / HEIGHT, (x1 - x0) / WIDTH, (y1 - y0) / HEIGHT))
    a.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])
    a.patch.set_alpha(0)
    for spine in a.spines.values():
        spine.set_visible(False)
    return a


def guide(a):
    a.axvline(TRANSIENT_S, color=GUIDE, lw=0.35, alpha=0.45, zorder=0)


def panel(x0, y0, x1, y1, letter, name):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2", fc="none",
                                ec=FRAME, lw=0.5))
    ax.text(x0 + 1.4, y0 + 1.3, letter, fontsize=8, fontweight="bold", va="top")
    ax.text(x0 + 4.6, y0 + 1.75, name, fontsize=6, fontweight="bold", va="top")


def image(array, x0, y0, width):
    height = width * array.shape[0] / array.shape[1]
    ax.imshow(array, extent=(x0, x0 + width, y0 + height, y0), interpolation="lanczos", zorder=1)
    return height


def block_arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="simple,head_width=3.4,head_length=2.2,tail_width=1.3",
                                 fc=BLOCK, ec="none", mutation_scale=1.6, zorder=4))


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_width=0.9,head_length=1.6",
                                 mutation_scale=1, lw=0.5, color=INK, shrinkA=0, shrinkB=0))


def bracket(x0, x1, y0, y1):
    for side, tick in [(x0, 0.8), (x1, -0.8)]:
        ax.plot([side + tick, side, side, side + tick], [y0, y0, y1, y1], color=INK, lw=0.5)


# a: video frames
panel(0.3, 0.3, 19.5, 41.0, "a", "Video")
for row, label in enumerate(["1", "$t$", "$T$"]):
    top = 7.2 + row * 11.8
    height = image(frame, 4.8, top, 12.0)
    ax.text(2.4, top + height / 2, label, fontsize=6, ha="center", va="center")
    if row < 2:
        ax.text(10.8, top + height + 1.6, "$\\vdots$", fontsize=6, ha="center", va="center", color=NEUTRAL)
block_arrow(19.7, 20, 23.1, 20)

# b: regions on the first frame
panel(23.3, 0.3, 53.5, 41.0, "b", "Regions")
photo_x, photo_y, photo_w = 25.0, 7.0, 26.8
photo_h = image(frame, photo_x, photo_y, photo_w)
to_mm = photo_w / frame.shape[1]
def on_photo(u, v):
    return photo_x + u * to_mm, photo_y + v * to_mm


for (x0, y0, x1, y1), colour, style in [(TORSO_BOX, TORSO, (0, (2.2, 1.2))), (CHEST_BOX, CHEST, "-")]:
    ax.add_patch(Rectangle(on_photo(x0, y0), (x1 - x0) * to_mm, (y1 - y0) * to_mm, fc="none", ec=colour, lw=0.7,
                           ls=style, zorder=2))
shoulder_v = (CHEST_BOX[1] + CHEST_BOX[3]) / 2
for u in (CHEST_BOX[0], CHEST_BOX[2]):
    ax.add_patch(Circle(on_photo(u, shoulder_v), 0.6, fc="white", ec=INK, lw=0.5, zorder=3))
photo_bottom = photo_y + photo_h
for (x, y), label_y, label, colour in [(on_photo(80, TORSO_BOX[3]), photo_bottom + 2.8, "Torso box", TORSO),
                                       (on_photo(150, CHEST_BOX[3]), photo_bottom + 6.2, "Chest ROI", CHEST),
                                       (on_photo(CHEST_BOX[2], shoulder_v + 3), photo_bottom + 9.6,
                                        "Shoulder landmark", INK)]:
    ax.plot([x, x], [y, label_y - 1.1], color=NEUTRAL, lw=0.35, zorder=3)
    ax.text(x + (0.4 if colour == INK else 0), label_y, label, fontsize=5.5, va="center", color=INK,
            ha="right" if colour == INK else "center")
block_arrow(53.7, 20, 57.1, 20)

# c: the nine observations of the extraction, grouped by image measurement; the three of the bank are framed in ink
# and numbered as in panel d, the other six have a light frame. Each tile is the operator image of the submitted
# figure (TILES, pixel boxes in the nine-observation PNG), and each trace is that observation in this trial.
panel(57.3, 0.3, WIDTH - 0.3, 41.0, "c", "Video-derived observations")
ROUTES = [("Optical flow", CHEST, [("of_farneback", "OF", "of_farneback"),
                                    ("of_disp_bridge", "OF bridge", None)]),
          ("Frame difference", CHEST, [("DoF", "DoF", None), ("dof_disp_bridge", "DoF bridge", None)]),
          ("1-D intensity profile (P1D)", CHEST, [("profile1D linear", "Linear", None),
                                                  ("profile1D quadratic", "Quadratic", "profile1d_quadratic"),
                                                  ("profile1D cubic", "Cubic", None),
                                                  ("profile1d_consensus", "Consensus", None)]),
          ("Vertical shift", TORSO, [("torso_shift", "Torso\ndisplacement", "torso_shift")])]
TILES = {"of_farneback": (958, 138, 1182, 317), "of_disp_bridge": (1216, 140, 1441, 320),
         "DoF": (1475, 140, 1700, 318), "dof_disp_bridge": (1735, 140, 1960, 318),
         "profile1D linear": (1994, 134, 2219, 312), "profile1D quadratic": (2253, 134, 2478, 312),
         "profile1D cubic": (2512, 134, 2737, 312), "profile1d_consensus": (2772, 132, 2997, 312),
         "torso_shift": (3031, 134, 3256, 310)}
INSET = 7
left, right, inner_gap, route_gap = 59.2, WIDTH - 2.0, 1.1, 3.0
n_tiles = sum(len(r[2]) for r in ROUTES)
slot = (right - left - inner_gap * (n_tiles - len(ROUTES)) - route_gap * (len(ROUTES) - 1)) / n_tiles
x = left
for route, colour, members in ROUTES:
    route_start = x
    for name, label, family in members:
        in_bank = family is not None
        middle = x + slot / 2
        ax.text(middle, 8.6, label, fontsize=5.5, ha="center", va="center", linespacing=0.95, color=INK,
                fontweight="bold" if in_bank else "normal")
        x0, y0, x1, y1 = TILES[name]
        tile = figure[y0 + INSET:y1 - INSET, x0 + INSET:x1 - INSET]
        height = image(tile, x, 11.4, slot)
        ax.add_patch(Rectangle((x, 11.4), slot, height, fc="none", ec=INK if in_bank else FRAME,
                               lw=1.0 if in_bank else 0.4, zorder=2))
        z = median_z(band_passed(signals[name]))
        trace = axes_at(x, 11.4 + height + 0.6, x + slot, 11.4 + height + 7.2)
        guide(trace)
        draw_trace(trace, *shown(z), OBS, lw=0.5 if in_bank else 0.35)
        if in_bank:
            k = families.index(family)
            ax.text(middle, 30.6, f"$o_{k + 1}(t)$", fontsize=6.5, ha="center", va="center")
        if name == "of_disp_bridge":
            scale = (x, x + slot * 10 / (SPAN_S[1] - SPAN_S[0]))
        x += slot + inner_gap
    x += route_gap - inner_gap
    route_end = x - route_gap
    # the bracket carries the region's colour only when one of its observations is in the bank; names stay in ink
    active = any(family is not None for _, _, family in members)
    ax.plot([route_start, route_start, route_end, route_end], [33.3, 34.2, 34.2, 33.3], color=colour, lw=0.6)
    ax.text((route_start + route_end) / 2, 36.4, route, fontsize=5.5, ha="center", va="center", color=INK)
# time scale of the traces: a 10-s double arrow under the OF bridge trace, which carries no o_k label
ax.add_patch(FancyArrowPatch((scale[0], 30.6), (scale[1], 30.6), arrowstyle="<|-|>,head_width=0.7,head_length=1.0",
                             mutation_scale=1, lw=0.45, color=INK, shrinkA=0, shrinkB=0))
ax.text(scale[1] + 0.8, 30.6, "10 s", fontsize=5, va="center")
# what the frame means, in the title row of the panel
ax.add_patch(Rectangle((right - 33.0, 2.1), 3.2, 2.2, fc="none", ec=INK, lw=1.0))
ax.text(right - 29.0, 3.2, "used by the filter, $o_1$–$o_3$", fontsize=5.5, va="center")
block_arrow(64.0, 41.2, 64.0, 44.6)

# d: observation law; each row of the observation vector, its noise variance, its trust, and its update share
panel(0.3, 44.8, 72.0, HEIGHT - 0.3, "d", "Reliability-weighted observation law")
rows_y = [58.0, 63.2, 68.4]
header_y = 52.4
for x, text in [(13.0, "Observations"), (33.0, "Noise variances"), (45.6, "Trust"), (60.3, "Update share $s_{k,t}$")]:
    ax.text(x, header_y, text, fontsize=5.5, ha="center", va="center")
bracket(8.2, 17.8, 55.2, 71.2)
bracket(27.6, 38.4, 55.2, 71.2)
ax.text(4.6, 63.2, "$\\mathbf{y}_t$ =", fontsize=7, ha="center", va="center")
for k, (y, family) in enumerate(zip(rows_y, families)):
    ax.text(13.0, y, f"$o_{k + 1}(t)$", fontsize=7, ha="center", va="center")
    ax.text(33.0, y, f"$R^{{\\mathrm{{eff}}}}_{{{k + 1},t}}$", fontsize=7, ha="center", va="center")
    ax.text(45.6, y, f"$\\pi_{{{k + 1},t}}$", fontsize=7, ha="center", va="center")
    arrow(43.2, y, 39.6, y)
    strip = axes_at(50.6, y - 2.1, 70.0, y + 2.1, ylim=(0, 1))
    guide(strip)
    time, share = shown(update_share[k])
    strip.fill_between(time, 0, share, color=tint(FAMILIES[family][2]), lw=0)
    strip.axhline(0, color=NEUTRAL, lw=0.3)
    if k == 0:
        strip.plot([SPAN_S[0]] * 2, [0, 1], color=NEUTRAL, lw=0.3, clip_on=False)
        for value in (0, 1):
            strip.text(SPAN_S[0] - 0.8, value, str(value), fontsize=5, ha="right", va="center")
ax.text(36.2, 77.6, "$R^{\\mathrm{eff}}_{k,t} \\propto R^{\\mathrm{self}}_{k,t}\\,/\\,\\pi_{k,t}$,      "
        "$s_{k,t} = (1/R^{\\mathrm{eff}}_{k,t})\\,/\\,\\Sigma_j\\,(1/R^{\\mathrm{eff}}_{j,t})$",
        fontsize=6.5, ha="center", va="center")
block_arrow(72.2, 63.2, 75.4, 63.2)

# e: oscillator state-space model and the smoothed state components
panel(75.6, 44.8, 129.0, HEIGHT - 0.3, "e", "State-space model")
centres = [88.0, 102.0, 116.0]
node_y, out_y, NODE = 53.2, 61.0, 3.2
for x, sub in zip(centres, ["t-1", "t", "t+1"]):
    ax.add_patch(Circle((x, node_y), NODE, fc="white", ec=INK, lw=0.5))
    ax.text(x, node_y, f"$\\mathbf{{x}}_{{{sub}}}$", fontsize=6, ha="center", va="center")
    arrow(x, node_y + NODE, x, out_y - 1.8)
    ax.text(x, out_y, f"$\\mathbf{{y}}_{{{sub}}}$", fontsize=6.5, ha="center", va="center")
for x0, x1 in zip(centres[:-1], centres[1:]):
    arrow(x0 + NODE, node_y, x1 - NODE, node_y)
    ax.text((x0 + x1) / 2, node_y - 0.9, "$F_t$", fontsize=6, ha="center", va="bottom")
for x in (81.0, 123.0):
    ax.text(x, node_y, "$\\cdots$", fontsize=6, ha="center", va="center")
ax.text(centres[1] + 0.9, (node_y + NODE + out_y - 1.8) / 2, "$H_t$", fontsize=6, ha="left", va="center")
decomposition = estimate["decomposition"]
fundamental_scale = 1.4826 * np.median(np.abs(np.asarray(decomposition["h1"], float)))
ax.text(78.0, 64.6, "Smoothed components of $\\mathbf{x}_t$", fontsize=5.5, va="center")
for row, (key, label, gain) in enumerate(COMPONENTS):
    y = 68.6 + row * 3.8
    ax.text(78.0, y, label, fontsize=5.5, va="center")
    component = axes_at(96.5, y - 1.8, 127.6, y + 1.8, ylim=(-2.2, 2.2))
    guide(component)
    time, values = shown(gain * np.asarray(decomposition[key], float) / fundamental_scale)
    component.plot(time, values, color=PARH if key == "h1" else INK, lw=0.5)
block_arrow(129.2, 63.2, 132.4, 63.2)

# f: outputs
panel(132.6, 44.8, WIDTH - 0.3, HEIGHT - 0.3, "f", "Outputs")
rate = fig.add_axes((143.5 / WIDTH, 1 - 61.0 / HEIGHT, 30.0 / WIDTH, 9.6 / HEIGHT))
rate.plot(windows.window + 15, windows.parh_track_bpm, color=PARH, lw=0.6)
low, high = np.floor(windows.parh_track_bpm.min()) - 1, np.ceil(windows.parh_track_bpm.max()) + 1
rate.set(xlim=SPAN_S, xticks=[10, 20, 30, 40, 50], xticklabels=[], ylim=(low, high), yticks=[low, high])
rate.set_ylabel("Rate (bpm)", fontsize=5.5)
waveform = fig.add_axes((143.5 / WIDTH, 1 - 75.2 / HEIGHT, 30.0 / WIDTH, 9.6 / HEIGHT))
draw_trace(waveform, *shown(median_z(np.asarray(estimate["z_full_smoothed"], float))), PARH, lw=0.5)
waveform.set(xlim=SPAN_S, xticks=[10, 20, 30, 40, 50], ylim=(-3.3, 3.3), yticks=[])
waveform.set_ylabel("Waveform", fontsize=5.5)
waveform.set_xlabel("Time (s)", fontsize=5.5, labelpad=1)
for axes in (rate, waveform):
    axes.tick_params(labelsize=5.5, pad=1, length=1.5, width=0.4)
    axes.yaxis.set_label_coords(-0.16, 0.5)
    for spine in axes.spines.values():
        spine.set_linewidth(0.4)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    guide(axes)
rate.text(TRANSIENT_S + 0.8, high - 0.06 * (high - low), "Transient", fontsize=5, color=NEUTRAL, va="top")

fig.savefig(out_pdf, dpi=600)
print(f"{trial}: families {families}; transient {TRANSIENT_S:.1f} s; "
      f"rate {windows.parh_track_bpm.min():.1f}-{windows.parh_track_bpm.max():.1f} bpm")
