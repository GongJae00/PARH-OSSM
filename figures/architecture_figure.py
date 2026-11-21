"""Figure 1: PARH-OSSM as a stack of stages on one time axis, drawn at the text width of the article (175.9 mm) so
that LaTeX does not scale it.

Left column. (a) Where the observations are measured: breathing moves the upper torso vertically; the chest strip
sits on the shoulder line of the first frame (as wide as the shoulder span, 0.2 span high) and the torso box below
the face. The body is an outline, not a photograph. (c) The state-space model; the node x_t is joined to the
stack of smoothed state components on the right.
Right column, one stack on one time axis. (b) The observations of one trial, band-passed to 0.08-0.5 Hz and displayed
as in Figure 2 (figure_style.draw_trace), each over its share of the Kalman update, w_k = (1 / R_eff,k) /
sum_j (1 / R_eff,j), from R_eff_t_channels of the run; the share, not the trust pi alone, is what weights an
observation, because a transient that breaks an observation raises its innovation variance and so its R_eff.
(c) The RTS-smoothed state components (decomposition of the run), on the scale of the fundamental. (d) The rate
track (window rates, plotted at the window centre) and the smoothed waveform. One light guide marks the transient,
the time of the largest median over the observations of |z|. Every trace is the stored output of the bank's run;
nothing is fitted here.

usage: architecture_figure.py <observation pkl> <final pkl> <window rates csv> <out.pdf>
"""
import json
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path
from scipy.signal import butter, sosfiltfilt

matplotlib.use("pdf")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (CHEST, GUIDE, INK, MM, NEUTRAL, OBS, PARH, TORSO, draw_share, draw_trace, lighter,  # noqa: E402
                          median_z)

plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                     "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold"})

observation_pkl, final_pkl, rates_csv, out_pdf = sys.argv[1:5]
WIDTH = 175.9
SPAN_S, BAND_HZ, TICKS_S = (5.0, 50.0), (0.08, 0.5), [10, 20, 30, 40, 50]
OUTLINE = "#b5b5b5"
# observation family of the run -> (label, label in the model diagram, colour of the region it reads)
FAMILIES = {"of_farneback": ("OF", "OF", CHEST), "profile1d_quadratic": ("P1D", "P1D", CHEST),
            "torso_shift": ("Torso displacement", "Torso disp.", TORSO)}
SIGNAL_OF = {"of_farneback": "of_farneback", "profile1d_quadratic": "profile1D quadratic",
             "torso_shift": "torso_shift"}
COMPONENTS = [("h1", "Fundamental", 1), ("h2", "Second harmonic", 1), ("baseline", "Baseline", 1),
              ("residual", "Residual (×10)", 10)]

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
    time = np.arange(x.size) / fps
    keep = (time >= SPAN_S[0]) & (time <= SPAN_S[1])
    return time[keep], x[keep]


scaled = {family: median_z(band_passed(signals[SIGNAL_OF[family]])) for family in families}
# the transient: the time of the largest median over the observations of |z|
n_common = min(z.size for z in scaled.values())
median_abs = np.median([np.abs(z[:n_common]) for z in scaled.values()], axis=0)
common_time = np.arange(n_common) / fps
in_span = (common_time >= SPAN_S[0]) & (common_time <= SPAN_S[1])
TRANSIENT_S = float(common_time[np.argmax(np.where(in_span, median_abs, -1))])

# vertical layout of the right column, in mm
LEFT_X, LABEL_X, PLOT_X0, PLOT_X1 = 60.0, 63.0, 86.0, 174.5
TEXT_LINE = 3.8
B_TOP, LANE, TRACE_H, SHARE_H = 8.0, 11.0, 7.0, 2.6
B_END = B_TOP + len(families) * LANE - (LANE - TRACE_H - 0.4 - SHARE_H)
C_TOP, C_LANE = B_END + TEXT_LINE, 5.2
C_END = C_TOP + len(COMPONENTS) * C_LANE - 0.6
D_HEAD = C_END + TEXT_LINE + 2.6
RATE_Y0, WAVE_Y0, WAVE_Y1 = D_HEAD + 3.0, D_HEAD + 8.4, D_HEAD + 18.4
HEIGHT = WAVE_Y1 + 8.6

fig = plt.figure(figsize=(WIDTH * MM, HEIGHT * MM))
canvas = fig.add_axes((0, 0, 1, 1))
canvas.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
canvas.axis("off")


def axes_mm(x0, y0, x1, y1):
    ax = fig.add_axes((x0 / WIDTH, 1 - y1 / HEIGHT, (x1 - x0) / WIDTH, (y1 - y0) / HEIGHT))
    ax.set_xlim(*SPAN_S)
    ax.patch.set_visible(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set(xticks=[], yticks=[])
    return ax


def heading(x, y, letter, name):
    canvas.text(x, y, letter, fontsize=8, fontweight="bold", va="baseline")
    canvas.text(x + 3.4, y, name, fontsize=6.5, fontweight="bold", va="baseline")


def arrow(p0, p1, lw=0.5, head=(0.8, 1.3)):
    canvas.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_width={head[0]},head_length={head[1]}",
                                     mutation_scale=1, lw=lw, color=INK, shrinkA=0, shrinkB=0))


def row_label(y, text):
    canvas.text(LABEL_X, y, text, fontsize=5.5, va="center")


def time_to_mm(t):
    return PLOT_X0 + (t - SPAN_S[0]) / (SPAN_S[1] - SPAN_S[0]) * (PLOT_X1 - PLOT_X0)


# the guide at the transient runs behind every row of the right column
canvas.plot([time_to_mm(TRANSIENT_S)] * 2, [B_TOP - 1.0, WAVE_Y1], color="#d6d6d6", lw=0.6, zorder=0)

# a: where the observations are measured
heading(0.5, 4.0, "a", "Measurement regions")
cx, top = 20.0, 9.0
canvas.add_patch(Ellipse((cx, top + 6.0), 8.6, 10.6, fc="none", ec=OUTLINE, lw=0.7))
body = Path([(cx - 3.0, top + 11.0), (cx - 3.0, top + 13.4), (cx - 15.0, top + 15.2), (cx - 18.6, top + 18.4),
             (cx - 19.6, top + 30.0), (cx + 19.6, top + 30.0), (cx + 18.6, top + 18.4), (cx + 15.0, top + 15.2),
             (cx + 3.0, top + 13.4), (cx + 3.0, top + 11.0)],
            [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.MOVETO, Path.LINETO,
             Path.CURVE3, Path.CURVE3, Path.LINETO])
canvas.add_patch(PathPatch(body, fc="none", ec=OUTLINE, lw=0.7))
shoulder_y, span = top + 15.6, 27.0
strip_height = 0.2 * span
canvas.add_patch(Rectangle((cx - span / 2, shoulder_y - strip_height / 2), span, strip_height, fc="none", ec=CHEST,
                           lw=0.8))
canvas.add_patch(Rectangle((cx - 9.5, top + 12.0), 19.0, 18.0, fc="none", ec=TORSO, lw=0.8, ls=(0, (2.2, 1.2))))
for x in (cx - span / 2, cx + span / 2):
    canvas.add_patch(Circle((x, shoulder_y), 0.75, fc="white", ec=INK, lw=0.5, zorder=3))
# breathing motion, inside the torso box
canvas.add_patch(FancyArrowPatch((cx - 5.4, top + 19.6), (cx - 5.4, top + 27.0),
                                 arrowstyle="<|-|>,head_width=0.8,head_length=1.3", mutation_scale=1, lw=0.5,
                                 color=INK, shrinkA=0, shrinkB=0))
canvas.text(cx - 3.8, top + 23.3, "Breathing\nmotion", fontsize=5.5, va="center", linespacing=1.1)
# direct labels beside the marks, each joined to its mark by a hairline
LABEL_A = 42.5
for (x0, y0), (x1, y1), text, colour in [
        ((cx + span / 2 + 0.7, shoulder_y - 0.4), (LABEL_A - 1.0, shoulder_y - 5.6), "Shoulder landmark", INK),
        ((cx + span / 2, shoulder_y + strip_height / 2), (LABEL_A - 1.0, shoulder_y + strip_height / 2), "Chest strip", CHEST),
        ((cx + 9.5, top + 28.4), (LABEL_A - 1.0, top + 28.4), "Torso box", TORSO)]:
    canvas.plot([x0, x1], [y0, y1], color=GUIDE, lw=0.35)
    canvas.text(LABEL_A, y1, text, fontsize=5.5, va="center", color=colour)

# b: observations over their share of the update
heading(LEFT_X, 4.0, "b", "Observations and their share of the update")
for k, family in enumerate(families):
    label, _, colour = FAMILIES[family]
    y0 = B_TOP + k * LANE
    trace_axes = axes_mm(PLOT_X0, y0, PLOT_X1, y0 + TRACE_H)
    time, z = shown(scaled[family])
    beyond = draw_trace(trace_axes, time, z, OBS, lw=0.45)
    trace_axes.set_ylim(-3.2, 3.2)
    if k == 0 and beyond.any():
        # the first compressed stretch is named once, in its own tone, beside its end
        # named beside the plateau: right of the end of the compressed stretch that holds the largest value
        peak = int(np.argmax(np.where(beyond, np.abs(z), -1)))
        end = peak + int(np.argmin(beyond[peak:]))
        trace_axes.text(time[end] + 0.5, 3.0, "compressed", fontsize=5, color=lighter(OBS, 0.3), va="top")
    share_y = y0 + TRACE_H + 0.4
    share_axes = axes_mm(PLOT_X0, share_y, PLOT_X1, share_y + SHARE_H)
    share_time, share = shown(update_share[k])
    draw_share(share_axes, share_time, share, colour)
    row_label(y0 + TRACE_H / 2, label)
    if k == 0:
        row_label(share_y + SHARE_H / 2, "Update share")
        canvas.plot([PLOT_X0 - 1.2] * 2, [share_y, share_y + SHARE_H], color=INK, lw=0.4)
        for value, y in ((1, share_y), (0, share_y + SHARE_H)):
            canvas.text(PLOT_X0 - 2.0, y, str(value), fontsize=4.5, ha="right", va="center")

# c: the state-space model (left) and its smoothed state components (right)
heading(0.5, C_TOP + 2.0, "c", "State-space model and latent state")
state_y = C_TOP + 13.0
observation_y = state_y + 19.0
for x, name in [(9.0, "$\\mathbf{x}_{t-1}$"), (25.0, "$\\mathbf{x}_t$"), (41.0, "$\\mathbf{x}_{t+1}$")]:
    canvas.add_patch(Circle((x, state_y), 3.4, fc="white", ec=INK, lw=0.6))
    canvas.text(x, state_y, name, fontsize=6, ha="center", va="center")
for x0 in (9.0, 25.0):
    arrow((x0 + 3.4, state_y), (x0 + 12.6, state_y))
    canvas.text(x0 + 8.0, state_y - 1.6, "$F_t$", fontsize=5.5, ha="center", va="baseline")
for k, family in enumerate(families):
    _, label, colour = FAMILIES[family]
    x = 25.0 + (k - (len(families) - 1) / 2) * 14.0
    canvas.add_patch(FancyBboxPatch((x - 6.0, observation_y - 2.0), 12.0, 4.0,
                                    boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=colour, lw=0.7))
    canvas.text(x, observation_y, label, fontsize=5.5, ha="center", va="center")
    arrow((25.0, state_y + 3.4), (x, observation_y - 2.0))
canvas.text(26.6, state_y + 8.2, "$H_t$", fontsize=5.5, va="center")
canvas.text(1.0, observation_y + 7.4, "$\\mathbf{x}_t = F_t\\mathbf{x}_{t-1} + \\mathbf{w}_t$,  "
            "$\\mathbf{y}_t = H_t\\mathbf{x}_t + \\mathbf{v}_t$", fontsize=6, va="center")
canvas.text(1.0, observation_y + 11.4, "$\\mathbf{v}_t \\sim \\mathcal{N}(\\mathbf{0},\\,\\mathrm{diag}\\,R^{\\mathrm{eff}}_{k,t})$,  "
            "$R^{\\mathrm{eff}}_{k,t} \\propto R^{\\mathrm{self}}_{k,t}\\,/\\,\\pi_{k,t}$", fontsize=6, va="center")
decomposition = estimate["decomposition"]
fundamental_scale = 1.4826 * np.median(np.abs(np.asarray(decomposition["h1"], float)))
for k, (key, label, gain) in enumerate(COMPONENTS):
    y0 = C_TOP + k * C_LANE
    ax = axes_mm(PLOT_X0, y0, PLOT_X1, y0 + C_LANE - 0.6)
    time, values = shown(gain * np.asarray(decomposition[key], float) / fundamental_scale)
    ax.plot(time, values, color=PARH if key == "h1" else OBS, lw=0.5)
    ax.set_ylim(-2.2, 2.2)
    row_label(y0 + (C_LANE - 0.6) / 2, label)
# one connector from the node x_t to the stack of state components, which a bracket spans
bracket_x = LEFT_X + 0.8
canvas.plot([bracket_x] * 2, [C_TOP, C_END], color=INK, lw=0.5)
connector_y = C_TOP + 5.2
canvas.plot([25.0, 25.0], [state_y - 3.4, connector_y], color=INK, lw=0.5)
arrow((25.0, connector_y), (bracket_x, connector_y))

# d: outputs
heading(LEFT_X, D_HEAD, "d", "Outputs")
rate = axes_mm(PLOT_X0, RATE_Y0, PLOT_X1, RATE_Y0 + 4.6)
rate.plot(windows.window + 15, windows.parh_track_bpm, color=PARH, lw=0.8)
low, high = np.floor(windows.parh_track_bpm.min()) - 1, np.ceil(windows.parh_track_bpm.max()) + 1
rate.set(ylim=(low, high), yticks=[low, high])
rate.spines["left"].set_visible(True)
rate.tick_params(length=1.2, pad=0.8, labelsize=5)
row_label(RATE_Y0 + 2.3, "Rate (bpm)")
waveform = axes_mm(PLOT_X0, WAVE_Y0, PLOT_X1, WAVE_Y1)
time, z = shown(median_z(np.asarray(estimate["z_full_smoothed"], float)))
draw_trace(waveform, time, z, PARH, lw=0.5)
waveform.set(ylim=(-3.4, 3.4), xticks=TICKS_S)
waveform.spines["bottom"].set_visible(True)
waveform.tick_params(length=1.5, pad=1, labelsize=5.5)
waveform.set_xlabel("Time (s)", fontsize=5.5, labelpad=1)
row_label(WAVE_Y0 + (WAVE_Y1 - WAVE_Y0) / 2, "Waveform")

fig.savefig(out_pdf, dpi=600)
print(f"{trial}: families {families}; update share {update_share.min():.2f}-{update_share.max():.2f}; "
      f"rate {windows.parh_track_bpm.min():.1f}-{windows.parh_track_bpm.max():.1f} bpm; "
      f"transient {TRANSIENT_S:.1f} s; height {HEIGHT:.1f} mm")
