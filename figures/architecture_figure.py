"""Figure 1: the pipeline of PARH-OSSM, drawn at the text width of the article (175.9 mm) so that LaTeX does not
scale it.

The photographs and the observation thumbnails are raster crops of the nine-observation version of the submitted
figure (the frame of panel a is the frame of panel b, without the boxes drawn on it); the boxes, the shoulder points,
the traces, the diagrams, and all text are drawn here. Panel c shows the five observations of the bank, one for each
image measurement (Methods); P1D is drawn as what it computes: the cross-correlation of consecutive profiles at
integer lags, interpolated quadratically (scipy interp1d, as in observations/chest_motion.py), with the refined peak
marked. Panel b places the boxes as the
nine-observation figure placed them: the chest ROI centred on the shoulders and as wide as their span, the torso box
below the face down to the bottom of the frame. The traces under the thumbnails and the outputs are one COHFACE trial,
the observations band-passed to 0.08-0.5 Hz from 2 s on and scaled by their median absolute deviation as in Figure 2.
Panel d draws the observation law of Methods: the five observations form the observation vector, and the trust of
each observation divides its noise variance, so that it never weights a sum of the observations.

usage: architecture_figure.py <nine-observation figure png> <observation pkl> <final pkl> <window rates csv> <out.pdf>
"""
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt

matplotlib.use("pdf")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import INK, MM, NEUTRAL, PARH  # noqa: E402

plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                     "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold"})

source_png, observation_pkl, final_pkl, rates_csv, out_pdf = sys.argv[1:6]
WIDTH, HEIGHT = 175.9, 80.0
# (method, label, tile of the nine-observation figure); P1D is drawn, not cropped
OBSERVATIONS = [("of_farneback", "OF", 0), ("DoF", "DoF", 2), ("dof_disp_bridge", "DoF bridge", 3),
                ("profile1D quadratic", "P1D", None), ("torso_shift", "Torso displacement", 8)]
CHEST, TORSO, FRAME, BLOCK = "#d1342f", "#93479b", "#c8c8c8", "#a0a0a0"
TRACE_SPAN_S, BAND_HZ, SKIP_S = (5, 50), (0.08, 0.5), 2

# raster sources, in pixels of the nine-observation figure
figure = np.array(Image.open(source_png).convert("RGB"))
frame = figure[58:255, 74:348]
tiles = [(954, 1186, 135, 321), (1213, 1445, 138, 323), (1472, 1704, 137, 322), (1732, 1964, 138, 321),
         (1991, 2223, 131, 316), (2250, 2482, 132, 317), (2509, 2741, 132, 317), (2769, 3001, 129, 315),
         (3028, 3260, 129, 315)]                                         # x0, x1, y0, y1 of each tile frame
CHEST_BOX, TORSO_BOX = (66, 131, 210, 163), (52, 124, 219, 196)        # x0, y0, x1, y1 in frame pixels


def thumbnail(x0, x1, y0, y1, photograph=False):
    """The content of one observation tile without its frame, the cream fill around a drawn tile set to white."""
    tile = figure[y0 + 11:y1 - 11, x0 + 11:x1 - 11].astype(int)
    content = np.abs(tile - [253, 248, 238]).sum(2) > 30
    content &= np.abs(tile - [255, 255, 255]).sum(2) > 30
    rows, cols = np.nonzero(content)
    tile = tile[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    edge = lambda line: ((line.min(-1) > 150) & (line.min(-1) <= 228)).mean() > 0.3   # a light canvas border
    while not photograph and edge(tile[0]):
        tile = tile[1:]
    while not photograph and edge(tile[-1]):
        tile = tile[:-1]
    if not photograph:
        tile[(np.abs(tile - [253, 248, 238]).sum(2) < 30) | (tile.min(2) > 228)] = 255
    return tile.astype(np.uint8)


def correlation_icon(x0, y0, width, height, kind):
    """Cross-correlation samples at integer lags (grey), their interpolation (ink), and the refined peak."""
    lags = np.arange(-3, 4)
    samples = np.exp(-0.5 * ((lags - 0.42) / 1.1) ** 2)
    fine = np.linspace(-3, 3, 601)
    curve = interp1d(lags, samples, kind)(fine)
    icon = mini_axes(x0, y0, x0 + width, y0 + height)
    icon.plot(fine, curve, color=INK, lw=0.5)
    icon.plot(lags, samples, "o", ms=1.5, mfc="white", mec=NEUTRAL, mew=0.5)
    icon.plot([fine[curve.argmax()]] * 2, [-0.15, curve.max()], color=INK, lw=0.4)
    icon.set(xlim=(-3.4, 3.4), ylim=(-0.15, 1.18), xticks=[], yticks=[])
    for spine in icon.spines.values():
        spine.set(color=FRAME, lw=0.5)


def scaled_trace(x, fps):
    x = np.asarray(x, float).reshape(-1)[int(SKIP_S * fps):]
    x = sosfiltfilt(butter(2, BAND_HZ, btype="band", fs=fps, output="sos"), x)
    z = (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))
    time = SKIP_S + np.arange(x.size) / fps
    keep = (time >= TRACE_SPAN_S[0]) & (time <= TRACE_SPAN_S[1])
    return (time[keep] - TRACE_SPAN_S[0]) / (TRACE_SPAN_S[1] - TRACE_SPAN_S[0]), 3 * np.tanh(z[keep] / 3) / 3


observations = pickle.load(open(observation_pkl, "rb"))
fps = float(observations["fps"])
signals = {e["method"]: e["estimate"]["signal_hat"] for e in observations["estimates"]}
final = pickle.load(open(final_pkl, "rb"))
estimate = final["estimates"][0]["estimate"]
trial = final_pkl.rsplit("/", 1)[1][:-4]
windows = pd.read_csv(rates_csv).query("trial == @trial")

fig = plt.figure(figsize=(WIDTH * MM, HEIGHT * MM))
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
ax.axis("off")


def mini_axes(x0, y0, x1, y1):
    return fig.add_axes((x0 / WIDTH, 1 - y1 / HEIGHT, (x1 - x0) / WIDTH, (y1 - y0) / HEIGHT))


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
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="simple,head_width=2.4,head_length=1.5,tail_width=1.0",
                                 fc=BLOCK, ec="none", mutation_scale=1.6))


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_width=0.9,head_length=1.6",
                                 mutation_scale=1, lw=0.5, color=INK, shrinkA=0, shrinkB=0))


# a: video frames
panel(0.3, 0.3, 19.5, 41.0, "a", "Video")
for row, label in enumerate(["1", "$t$", "$T$"]):
    top = 7.2 + row * 11.8
    height = image(frame, 4.8, top, 12.0)
    ax.text(2.4, top + height / 2, label, fontsize=6, ha="center", va="center")
    if row < 2:
        ax.text(10.8, top + height + 1.6, "$\\vdots$", fontsize=6, ha="center", va="center", color=NEUTRAL)

# b: regions of interest
panel(23.3, 0.3, 53.5, 41.0, "b", "Regions")
photo_x, photo_y, photo_w = 25.0, 7.0, 26.8
photo_h = image(frame, photo_x, photo_y, photo_w)
to_mm = photo_w / frame.shape[1]
for (x0, y0, x1, y1), colour in [(TORSO_BOX, TORSO), (CHEST_BOX, CHEST)]:
    ax.add_patch(Rectangle((photo_x + x0 * to_mm, photo_y + y0 * to_mm), (x1 - x0) * to_mm, (y1 - y0) * to_mm,
                           fc="none", ec=colour, lw=0.7, ls=(0, (2.2, 1.2)), zorder=2))
shoulder_y = photo_y + (CHEST_BOX[1] + CHEST_BOX[3]) / 2 * to_mm
for x in (CHEST_BOX[0], CHEST_BOX[2]):
    ax.add_patch(Circle((photo_x + x * to_mm, shoulder_y), 0.6, fc="white", ec=INK, lw=0.5, zorder=3))
key_y = photo_y + photo_h + 3.4
ax.add_patch(Circle((26.4, key_y), 0.6, fc="white", ec=INK, lw=0.5))
ax.text(28.6, key_y, "Shoulder landmarks", fontsize=5.5, va="center")
for row, (colour, label) in enumerate([(CHEST, "Chest ROI"), (TORSO, "Torso box")]):
    y = key_y + 3.6 * (row + 1)
    ax.plot([25.3, 27.6], [y, y], color=colour, lw=0.7, ls=(0, (2.2, 1.2)))
    ax.text(28.6, y, label, fontsize=5.5, va="center")
block_arrow(19.7, 20, 23.1, 20)
block_arrow(53.7, 20, 57.1, 20)

# c: the five observations, each in a slot with its tile centred and its trace across the slot
panel(57.3, 0.3, WIDTH - 0.3, 41.0, "c", "Video-derived observations")
slot, gap, left = 21.16, 2.4, 58.7
pitch = slot + gap
for k, (name, label, tile) in enumerate(OBSERVATIONS):
    x = left + k * pitch
    middle = x + slot / 2
    ax.text(middle, 11.0, label, fontsize=5.5, ha="center", va="bottom")
    if tile is None:
        correlation_icon(middle - 5.5, 12.4, 11.0, 8.6, "quadratic")
    else:
        thumb = thumbnail(*tiles[tile], photograph=name == "torso_shift")
        thumb_h = 11.8 * thumb.shape[0] / thumb.shape[1]
        image(thumb, middle - 5.9, 11.8 + (9.8 - thumb_h) / 2, 11.8)
    t, y = scaled_trace(signals[name], fps)
    ax.plot(x + 0.75 + t * (slot - 1.5), 26.0 - y * 2.6, color=NEUTRAL, lw=0.45)
    ax.text(middle, 32.0, f"$o_{k + 1}(t)$", fontsize=7, ha="center", va="center")
for x0, x1, label, colour in [(left, left + 3 * pitch + slot, "Chest ROI", CHEST),
                              (left + 4 * pitch, left + 4 * pitch + slot, "Torso box", TORSO)]:
    ax.plot([x0, x1], [35.6, 35.6], color=colour, lw=0.7, ls=(0, (2.2, 1.2)))
    ax.text((x0 + x1) / 2, 37.8, label, fontsize=5.5, ha="center", va="center")
block_arrow(59.95, 41.2, 59.95, 44.6)

# d: observation law
panel(0.3, 44.8, 62.6, HEIGHT - 0.3, "d", "Reliability-weighted observation law")
rows_y = [57.4, 61.7, 65.6, 69.6]
entries = [("$o_1(t)$", "$R^{\\mathrm{eff}}_{1,t}$", "$\\pi_{1,t}$"), ("$o_2(t)$", "$R^{\\mathrm{eff}}_{2,t}$",
           "$\\pi_{2,t}$"), ("$\\vdots$", "$\\vdots$", "$\\vdots$"),
           ("$o_5(t)$", "$R^{\\mathrm{eff}}_{5,t}$", "$\\pi_{5,t}$")]
for x0, x1 in [(9.0, 19.0), (33.0, 43.0)]:
    for side, tick in [(x0, 0.8), (x1, -0.8)]:
        ax.plot([side + tick, side, side, side + tick], [55.0, 55.0, 72.0, 72.0], color=INK, lw=0.5)
ax.text(14.0, 52.2, "Observations", fontsize=5.5, ha="center", va="center")
ax.text(38.0, 52.2, "Noise variances", fontsize=5.5, ha="center", va="center")
ax.text(55.0, 52.2, "Trust", fontsize=5.5, ha="center", va="center")
ax.text(5.6, 63.5, "$\\mathbf{y}_t$ =", fontsize=7, ha="center", va="center")
for y, (observation, variance, trust) in zip(rows_y, entries):
    ax.text(14.0, y, observation, fontsize=7, ha="center", va="center")
    ax.text(38.0, y, variance, fontsize=7, ha="center", va="center")
    ax.text(55.0, y, trust, fontsize=7, ha="center", va="center")
    if trust != "$\\vdots$":
        arrow(52.2, y, 44.0, y)
ax.text(31.4, 76.0, "$\\mathbf{y}_t = H_t\\,\\mathbf{x}_t + \\mathbf{v}_t$,   "
        "$\\mathbf{v}_t \\sim \\mathcal{N}(\\mathbf{0},\\, \\mathrm{diag}\\,R^{\\mathrm{eff}}_{k,t})$,   "
        "$R^{\\mathrm{eff}}_{k,t} \\propto R^{\\mathrm{self}}_{k,t}\\,/\\,\\pi_{k,t}$",
        fontsize=6.5, ha="center", va="center")
block_arrow(62.8, 62, 66.2, 62)

# e: state-space model
panel(66.4, 44.8, 118.4, HEIGHT - 0.3, "e", "State-space model")
centres = [77.4, 92.4, 107.4]
for x, sub in zip(centres, ["t-1", "t", "t+1"]):
    ax.add_patch(Circle((x, 56.4), 3.1, fc="white", ec=INK, lw=0.5))
    ax.text(x, 56.4, f"$\\mathbf{{x}}_{{{sub}}}$", fontsize=7, ha="center", va="center")
    arrow(x, 59.5, x, 63.4)
    ax.text(x, 65.8, f"$\\mathbf{{y}}_{{{sub}}}$", fontsize=7, ha="center", va="center")
for x0, x1 in zip(centres[:-1], centres[1:]):
    arrow(x0 + 3.1, 56.4, x1 - 3.1, 56.4)
for x in (70.6, 114.2):
    ax.text(x, 56.4, "$\\cdots$", fontsize=6, ha="center", va="center")
ax.text((centres[0] + centres[1]) / 2, 55.2, "$F_t$", fontsize=6.5, ha="center", va="bottom")
ax.text(centres[1] + 0.8, 61.4, "$H_t$", fontsize=6.5, ha="left", va="center")
state_key = [("$h^{(1)}_c,\\, h^{(1)}_s$", "fundamental", 80.6, 71.4), ("$h^{(2)}_c,\\, h^{(2)}_s$", "second harmonic", 80.6, 75.9),
             ("$b,\\, \\dot b$", "baseline", 103.0, 71.4), ("$r,\\, \\dot r$", "residual", 103.0, 75.9)]
for symbols, meaning, x, y in state_key:
    ax.text(x, y, symbols, fontsize=7, ha="right", va="center")
    ax.text(x + 1.0, y, meaning, fontsize=5.5, ha="left", va="center")
block_arrow(118.6, 62, 122.0, 62)

# f: outputs
panel(122.2, 44.8, WIDTH - 0.3, HEIGHT - 0.3, "f", "Outputs")


rate = mini_axes(133.0, 51.0, 172.0, 59.6)
rate.plot(windows.window + 15, windows.parh_track_bpm, color=PARH, lw=0.8)
rate.set(xlim=(0, 60), xticks=[0, 15, 30, 45, 60], xticklabels=[], ylim=(8, 16), yticks=[8, 12, 16])
rate.set_ylabel("Rate (bpm)", fontsize=5.5)
waveform = mini_axes(133.0, 63.0, 172.0, 71.6)
video_time = np.arange(np.asarray(estimate["z_full_smoothed"]).size) / fps
shown = video_time >= SKIP_S
z = np.asarray(estimate["z_full_smoothed"], float)[shown]
waveform.plot(video_time[shown], (z - np.median(z)) / (1.4826 * np.median(np.abs(z - np.median(z)))), color=PARH,
              lw=0.5)
waveform.set(xlim=(0, 60), xticks=[0, 15, 30, 45, 60], ylim=(-3.2, 3.2), yticks=[])
waveform.set_ylabel("Waveform", fontsize=5.5)
waveform.set_xlabel("Time (s)", fontsize=5.5, labelpad=1)
for axes in (rate, waveform):
    axes.tick_params(labelsize=5.5, pad=1)
for axes in (rate, waveform):
    axes.yaxis.set_label_coords(-0.12, 0.5)

fig.savefig(out_pdf, dpi=600)
print(f"{trial}: rate {windows.parh_track_bpm.min():.1f}-{windows.parh_track_bpm.max():.1f} bpm")
