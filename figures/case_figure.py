"""Figure 2: one trial per dataset followed through the stages of PARH-OSSM, against the belt.

Columns: a COHFACE trial and a MAHNOB-HCI development trial, each the trial whose PARH-OSSM MAE against the
Hilbert reference (scored as main_tables.py scores it) is closest to the median over that dataset's
development trials, so neither case is chosen for looking good and no test subject is read. Rows:
- the three observations of the bank and the belt, each band-passed to 0.08-0.5 Hz as the evaluation filters the
  belt, centred on its median and scaled by its median absolute deviation, so that the filter's start-up and the
  end of a belt record do not set the scale; for display the values are compressed as 3 tanh(z / 3) and scaled so
  that no trace leaves its row, and the samples more than 4 deviations from the median, where the compression
  flattens the trace, are drawn in a lighter tone (figure_style.draw_trace, as in Figure 1). The observations are
  filtered from 2 s on: their first frames carry the initialisation step of the extraction (in COHFACE up to 150
  times the later spread), which the filter would spread over several seconds, and the observation calibration of
  the method skips the same 2 s (OBS_CAL_SKIP_SEC in parh_ossm.py);
- the share of each observation in the Kalman update, w_k = (1 / R_eff,k) / sum_j (1 / R_eff,j) from the effective
  measurement-noise variances of the filter (diagnostics R_eff_t_channels), as in Figure 1, one row per
  observation on a 0-1 scale;
- the output waveform that the waveform metrics score (z_full_smoothed, as bounded_lag_waveform.py reads it) and
  the belt, scaled the same way, without any lag or sign correction;
- the window rates that the tables score: the Hilbert reference, PARH-OSSM, the single-observation OSSM-KF,
  and the constant predictor, plotted at the window centres on the same span of rate in both columns.
The three upper rows of the long MAHNOB-HCI trial show a window of WINDOW_S = 60 s centred on the window centre
of the largest Hilbert reference rate, moved to lie inside the trial; the rate row keeps the whole trial and marks
that window with a light band. The rule uses the reference only to place the window, and nothing is fitted.

usage: case_figure.py <COHFACE window rates csv> <COHFACE final dir> <COHFACE observation dir>
                      <MAHNOB-HCI window rates csv> <MAHNOB-HCI final dir> <MAHNOB-HCI observation dir>
                      <split csv> <out.pdf>
"""
import json
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "estimate", "bank"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.reference.belt_reference import filter_RW, recorded_belt
from bank import BANK
from figure_style import CHEST, INK, MM, NEUTRAL, OBS, OSSM, PARH, TORSO, draw_share, draw_trace, lighter, median_z

matplotlib.use("pdf")
WIN, MIN_WINDOWS, MIN_HZ, MAX_HZ = 30, 10, 0.08, 0.5
OBSERVATION_START_SEC = 2.0
# (signal in the observation file, family in the filter's channel order, label)
OBSERVATIONS = [(label, label.lower().replace(" ", "_"), name, TORSO if label == "torso_shift" else CHEST)
                for label, name in BANK.items()]
WINDOW_S = 60.0
LINE_MM = 2.3  # minimum spacing of direct labels, in mm


def median_case(windows, development):
    """The trial whose PARH-OSSM Hilbert MAE is closest to the median over the development trials."""
    windows = windows[windows.trial.isin(windows.dropna(subset=["direct_p1d_welch_bpm"]).trial.unique())]
    windows = windows[windows.trial.isin(development)]
    valid = windows.dropna(subset=["hilbert_reference_bpm", "parh_track_bpm"])
    valid = valid.groupby("trial").filter(lambda w: len(w) >= MIN_WINDOWS)
    mae = (valid.parh_track_bpm - valid.hilbert_reference_bpm).abs().groupby(valid.trial).mean()
    trial = (mae - mae.median()).abs().idxmin()
    constant = windows.groupby("trial").hilbert_reference_bpm.mean().median()
    return trial, mae[trial], mae.median(), len(mae), constant, windows[windows.trial == trial]


(cohface_csv, cohface_final, cohface_observations, mahnob_csv, mahnob_final, mahnob_observations,
 split_csv, out_path) = sys.argv[1:]
split = pd.read_csv(split_csv)
cases = []
for name, csv_path, final_dir, observation_dir in [("COHFACE", cohface_csv, cohface_final, cohface_observations),
                                                    ("MAHNOB-HCI", mahnob_csv, mahnob_final, mahnob_observations)]:
    windows = pd.read_csv(csv_path)
    development = (windows.trial.unique() if name == "COHFACE"
                   else split[split.split == "development"].trial.values)
    trial, trial_mae, median_mae, n_trials, constant, trial_windows = median_case(windows, development)
    final = pickle.load(open(os.path.join(final_dir, f"{trial}.pkl"), "rb"))
    observations = {e["method"]: np.asarray(e["estimate"]["signal_hat"], dtype=float).reshape(-1)
                    for e in pickle.load(open(os.path.join(observation_dir, f"{trial}.pkl"), "rb"))["estimates"]}
    cases.append(dict(name=name, trial=trial, mae=trial_mae, median=median_mae, n=n_trials, constant=constant,
                      windows=trial_windows, final=final, observations=observations))
    print(f"{name}: {trial}, PARH-OSSM Hilbert MAE {trial_mae:.3f} bpm, median over {n_trials} trials {median_mae:.3f}")

rate_columns = ["hilbert_reference_bpm", "parh_track_bpm", "ossm_kf_track_bpm"]
rate_limits = [(min(c["windows"][rate_columns].min().min(), c["constant"]),
                max(c["windows"][rate_columns].max().max(), c["constant"])) for c in cases]
rate_span = np.ceil(max(high - low for low, high in rate_limits)) + 1

# layout in mm; the figure keeps the text width of the article
WIDTH = 175.9
COLUMNS = [(27.0, 88.0), (101.0, 161.0)]
TEXT_LINE = 3.8
ROW1_Y, ROW1_LANE = 8.0, 7.5
ROW2_Y, SHARE_H, SHARE_GAP = ROW1_Y + 4 * ROW1_LANE + TEXT_LINE, 3.4, 0.8
ROW3_Y, ROW3_H = ROW2_Y + 3 * SHARE_H + 2 * SHARE_GAP + TEXT_LINE, 16.0
ROW4_Y, ROW4_H = ROW3_Y + ROW3_H + 6.8 + TEXT_LINE, 26.0  # tick labels and "Time (s)", then one text line
HEIGHT = ROW4_Y + ROW4_H + 8.5
LANE_UNITS, GAIN = 1.6, 0.26

fig = plt.figure(figsize=(WIDTH * MM, HEIGHT * MM))
canvas = fig.add_axes((0, 0, 1, 1))
canvas.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
canvas.axis("off")
canvas.set_zorder(5)  # text and marks above the plotting axes


def axes_mm(x0, y0, x1, y1, xlim):
    ax = fig.add_axes((x0 / WIDTH, 1 - y1 / HEIGHT, (x1 - x0) / WIDTH, (y1 - y0) / HEIGHT))
    ax.set_xlim(*xlim)
    ax.patch.set_visible(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set(xticks=[], yticks=[])
    return ax


def to_mm(ax, x, y):
    """Position of a data point on the canvas, in mm from the top left."""
    px, py = ax.transData.transform((x, y))
    return px / fig.dpi * 25.4, HEIGHT - py / fig.dpi * 25.4


def spread(ys, gap=LINE_MM):
    """Move label positions apart to at least gap, keeping their order and their mean."""
    order = np.argsort(ys)
    y = np.asarray(ys, float)[order]
    for _ in range(50):
        for k in range(1, len(y)):
            if y[k] - y[k - 1] < gap:
                shift = (gap - (y[k] - y[k - 1])) / 2
                y[k - 1] -= shift
                y[k] += shift
    out = np.empty_like(y)
    out[order] = y
    return out


def end_labels(items, x_mm=None):
    """Direct labels beside the ends of lines: items are (ax, x, y, text, colour); labels that would overlap are spread."""
    anchors = [to_mm(ax, x, y) for ax, x, y, _, _ in items]
    ys = spread([a[1] for a in anchors])
    for (ax, x, y, text, colour), (ax_mm, _), label_y in zip(items, anchors, ys):
        line_key(ax_mm + 0.9 if x_mm is None else x_mm, label_y, text, colour)


def line_key(x_mm, y_mm, text, colour):
    """A name in ink after a short stroke of its line's colour, so that colour is never the only cue."""
    canvas.plot([x_mm, x_mm + 1.3], [y_mm, y_mm], color=colour, lw=0.9, solid_capstyle="butt")
    canvas.text(x_mm + 1.8, y_mm, text, fontsize=5, color=INK, va="center")


def row_title(y, text):
    canvas.text(0.5, y, text, fontsize=6, va="center")


row_title(ROW1_Y + 3.2, "Belt and\nobservations")
row_title(ROW2_Y + 1.7, "Update share")
row_title(ROW3_Y + 2.0, "Waveform")
row_title(ROW4_Y + 2.0, "Rate (bpm)")
compressed_named = False

for column, case in enumerate(cases):
    x0, x1 = COLUMNS[column]
    estimate = case["final"]["estimates"][0]["estimate"]
    fps, fs_gt = float(case["final"]["fps"]), float(case["final"]["fs_gt"])
    belt = filter_RW(recorded_belt(case["final"]["gt"]), fs_gt, lo=MIN_HZ, hi=MAX_HZ).reshape(-1)
    n_video = np.asarray(estimate["signal_hat"]).size
    video_time = np.arange(n_video) / fps
    belt_time = np.arange(belt.size) / fs_gt
    shown = belt_time <= video_time[-1]
    trial_span = (0.0, float(video_time[-1]))
    if column == 1:
        windows = case["windows"]
        centre_of_maximum = float(windows.loc[windows.hilbert_reference_bpm.idxmax(), "window"] + WIN / 2)
        start = min(max(centre_of_maximum - WINDOW_S / 2, 0.0), trial_span[1] - WINDOW_S)
        span = (start, start + WINDOW_S)
        print(f"{case['name']}: window {span[0]:.1f}-{span[1]:.1f} s, centred on the maximum of the Hilbert reference "
              f"rate at {centre_of_maximum:.1f} s ({windows.hilbert_reference_bpm.max():.2f} bpm)")
    else:
        span = trial_span

    identifier = case["trial"].split("_")[1:]
    name = (f"{case['name']}, subject {identifier[0]}, trial {identifier[1]}" if case["name"] == "COHFACE"
            else f"{case['name']}, session {identifier[0]}")
    letter_x = 0.5 if column == 0 else x0 - 8.0
    canvas.text(letter_x, 4.2, "ab"[column], fontsize=8, fontweight="bold", va="baseline")
    canvas.text(letter_x + 4.0, 4.2, name, fontsize=6.5, fontweight="bold", va="baseline")

    # row 1: the belt and the three observations
    traces = axes_mm(x0, ROW1_Y, x1, ROW1_Y + 4 * ROW1_LANE, span)
    traces.set_ylim(-3 * LANE_UNITS - LANE_UNITS / 2, LANE_UNITS / 2)
    z_belt = median_z(belt[shown])
    draw_trace(traces, belt_time[shown], z_belt, INK, gain=GAIN, lw=0.5)
    first = int(OBSERVATION_START_SEC * fps)
    for row, (method, _, label, _) in enumerate(OBSERVATIONS, start=1):
        observation = case["observations"][method][first:n_video]
        observation = filter_RW(observation, fps, lo=MIN_HZ, hi=MAX_HZ).reshape(-1) if observation.std() > 0 else observation
        time = video_time[first:first + observation.size]
        z = median_z(observation)
        beyond = draw_trace(traces, time, z, OBS, offset=-row * LANE_UNITS, gain=GAIN, lw=0.45)
        inside = (time >= span[0]) & (time <= span[1])
        if not compressed_named and (beyond & inside).any():
            # named beside the light segments: right of the last compressed stretch of the first cluster
            # (stretches less than 5 s apart), at the top of the row
            visible = np.flatnonzero(beyond & inside)
            gaps = np.flatnonzero(np.diff(time[visible]) > 5.0)
            last = visible[gaps[0]] if gaps.size else visible[-1]
            traces.text(time[last] + 0.015 * (span[1] - span[0]), -row * LANE_UNITS + LANE_UNITS / 2 - 0.05,
                        "compressed", fontsize=5, color=lighter(OBS, 0.3), va="top")
            compressed_named = True
        if column == 0:
            canvas.text(x0 - 1.5, ROW1_Y + (row + 0.5) * ROW1_LANE, label, fontsize=5.5, ha="right", va="center")
    if column == 0:
        canvas.text(x0 - 1.5, ROW1_Y + 0.5 * ROW1_LANE, "Belt", fontsize=5.5, ha="right", va="center", color=INK)

    # row 2: the update share of each observation
    precision = 1.0 / np.asarray(estimate["diagnostics"]["R_eff_t_channels"], dtype=float)
    update_share = precision / precision.sum(axis=0)
    families = json.loads(estimate["meta"])["observation_families"]
    share_time = np.linspace(0, video_time[-1], update_share.shape[1])
    for row, (_, family, label, colour) in enumerate(OBSERVATIONS):
        y = ROW2_Y + row * (SHARE_H + SHARE_GAP)
        share_axes = axes_mm(x0, y, x1, y + SHARE_H, span)
        draw_share(share_axes, share_time, update_share[families.index(family)], colour)
        if column == 0:
            canvas.text(x0 - 1.5, y + SHARE_H / 2, label, fontsize=5.5, ha="right", va="center")
        if column == 1 and row == 0:
            canvas.plot([x1 + 1.4] * 2, [y, y + SHARE_H], color=INK, lw=0.4)
            canvas.text(x1 + 2.2, y, "1", fontsize=4.5, va="center")
            canvas.text(x1 + 2.2, y + SHARE_H, "0", fontsize=4.5, va="center")

    # row 3: the waveforms
    waveform = axes_mm(x0, ROW3_Y, x1, ROW3_Y + ROW3_H, span)
    waveform.set_ylim(-3.3, 3.3)
    z_smoothed = median_z(np.asarray(estimate["z_full_smoothed"], dtype=float))
    draw_trace(waveform, belt_time[shown], z_belt, INK, lw=0.6)
    draw_trace(waveform, video_time[:z_smoothed.size], z_smoothed, PARH, lw=0.6)
    waveform.spines["bottom"].set_visible(True)
    waveform.set_xticks(np.arange(np.ceil(span[0] / 10) * 10, span[1] + 1e-6, 10))
    waveform.tick_params(length=1.5, pad=1, labelsize=5)
    waveform.set_xlabel("Time (s)", fontsize=5.5, labelpad=1)
    if column == 0:
        # a's traces run to the gutter, so its two names sit above the right end of the axis
        line_key(x1 - 12.6, ROW3_Y - 1.2, "PARH-OSSM", PARH)
        line_key(x1 - 19.6, ROW3_Y - 1.2, "Belt", INK)
    else:
        ends = []
        for label_text, time, z, colour in (("Belt", belt_time[shown], z_belt, INK),
                                            ("PARH-OSSM", video_time[:z_smoothed.size], z_smoothed, PARH)):
            k = np.flatnonzero(time <= span[1])[-1]
            ends.append((waveform, span[1], float(np.clip(3 * np.tanh(z[k] / 3), -3.3, 3.3)), label_text, colour))
        end_labels(ends, x_mm=x1 + 0.9)

    # row 4: the window rates over the whole trial
    rate = axes_mm(x0, ROW4_Y, x1, ROW4_Y + ROW4_H, trial_span)
    middle = sum(rate_limits[column]) / 2
    rate.set_ylim(middle - rate_span / 2, middle + rate_span / 2)
    centre = case["windows"].window + WIN / 2
    if column == 1:
        rate.axvspan(*span, color="#e9e9e9", lw=0, zorder=0)
        canvas.text(to_mm(rate, span[0], 0)[0] + 0.3, ROW4_Y - 0.5, "Window shown above", fontsize=4.8,
                    color="#6f6f6f", va="bottom")
    rate.plot([centre.min(), centre.max()], [case["constant"]] * 2, color=NEUTRAL, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    for key, colour in (("hilbert_reference_bpm", INK), ("parh_track_bpm", PARH), ("ossm_kf_track_bpm", OSSM)):
        rate.plot(centre, case["windows"][key], color=colour, lw=1.0, zorder=2)
    for side in ("left", "bottom"):
        rate.spines[side].set_visible(True)
    rate.yaxis.set_major_locator(MultipleLocator(2))
    rate.set_xticks(np.arange(0, trial_span[1] + 1e-6, 10 if column == 0 else 20))
    rate.tick_params(length=1.5, pad=1, labelsize=5)
    rate.set_xlabel("Time (s)", fontsize=5.5, labelpad=1)
    last = case["windows"].iloc[-1]
    labels = [(rate, centre.iloc[-1], last[key], text, colour) for key, text, colour in (
        ("hilbert_reference_bpm", "Hilbert reference", INK), ("parh_track_bpm", "PARH-OSSM", PARH),
        ("ossm_kf_track_bpm", "OSSM-KF", OSSM))]
    labels.append((rate, centre.iloc[-1], case["constant"], "Constant predictor", "#6f6f6f"))
    end_labels(labels)

fig.savefig(out_path)
fig.savefig(out_path.removesuffix(".pdf") + ".png", dpi=300)
print(f"height {HEIGHT:.1f} mm")
