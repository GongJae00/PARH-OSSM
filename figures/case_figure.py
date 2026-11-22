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
from matplotlib.textpath import TextPath
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "estimate", "bank"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.reference.belt_reference import filter_RW, recorded_belt
from bank import BANK
from figure_style import CHEST, INK, MM, NEUTRAL, OSSM, PARH, TORSO, draw_share, draw_trace, median_z, tint

matplotlib.use("pdf")
WIN, MIN_WINDOWS, MIN_HZ, MAX_HZ = 30, 10, 0.08, 0.5
OBSERVATION_START_SEC = 2.0
# (signal in the observation file, family in the filter's channel order, label)
OBSERVATIONS = [(label, label.lower().replace(" ", "_"), name, TORSO if label == "torso_shift" else CHEST)
                for label, name in BANK.items()]
WINDOW_S = 60.0


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

# layout in mm, at the text width of the article: one row per trial, one column per stage, as in the order in which
# the method uses them (observations, their update shares, the output waveform, the window rates)
WIDTH = 175.9
LABEL_X = 11.0
COLUMNS = {"observations": (12.0, 56.0), "share": (66.0, 94.0), "waveform": (100.0, 138.0), "rate": (149.0, 174.5)}
TITLES = {"observations": "Belt and observations", "share": "Update share", "waveform": "Waveform",
          "rate": "Rate (bpm)"}
PANEL_H, HEADER, TICKS, ROW_GAP = 26.0, 4.6, 7.4, 2.4
ROW_Y = [6.0 + HEADER, 6.0 + HEADER + PANEL_H + TICKS + ROW_GAP + HEADER]
HEIGHT = ROW_Y[1] + PANEL_H + TICKS + 2.0
LANE_UNITS, GAIN = 1.6, 0.26
SHARE_GAP = 2.0
SHARE_H = (PANEL_H - 1.0 - 2 * SHARE_GAP) / 3  # three lanes fill the panel height
SHORT = {"OF": "OF", "P1D": "P1D", "Torso displacement": "Torso"}

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


def time_axis(ax, span, step, label):
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.set_xticks(np.arange(np.ceil(span[0] / step) * step, span[1] + 1e-6, step))
    ax.tick_params(axis="x", length=1.5, pad=1, labelsize=5)
    if label:
        ax.set_xlabel("Time (s)", fontsize=5.5, labelpad=1)


def line_key(x_mm, y_mm, text, colour, dotted=False):
    """A name in ink after a short stroke of its line's colour, so that colour is never the only cue."""
    canvas.plot([x_mm, x_mm + 2.6], [y_mm, y_mm], color=colour, lw=0.9, solid_capstyle="butt",
                ls=(0, (1, 1.2)) if dotted else "-")
    return key_text(x_mm, y_mm, text)


def key_text(x_mm, y_mm, text):
    """The name of a key entry; returns where the next entry starts, a fixed gap after the measured text."""
    canvas.text(x_mm + 3.3, y_mm, text, fontsize=5.5, color=INK, va="center")
    width_pt = TextPath((0, 0), text, size=5.5).get_extents().width
    return x_mm + 3.3 + width_pt * 25.4 / 72 + 5.0


def area_key(x_mm, y_mm, text, colour):
    canvas.add_patch(plt.Rectangle((x_mm, y_mm - 0.9), 2.6, 1.8, color=tint(colour), lw=0))
    return key_text(x_mm, y_mm, text)


def entry_width(text):
    return 3.3 + TextPath((0, 0), text, size=5.5).get_extents().width * 25.4 / 72


def keys_in_line(x_right, y_mm, entries, gap=2.4):
    """Key entries on one line, ending at x_right: (text, colour, kind) with kind "line", "dotted", or "area"."""
    x = x_right - sum(entry_width(text) for text, _, _ in entries) - gap * (len(entries) - 1)
    for text, colour, kind in entries:
        (area_key if kind == "area" else line_key)(x, y_mm, text, colour, **({"dotted": True} if kind == "dotted" else {}))
        x += entry_width(text) + gap


def keys_stacked(x_right, y_mm, entries, step=2.3):
    """Key entries one per line, left-aligned so that the longest ends at x_right."""
    x = x_right - max(entry_width(text) for text, _, _ in entries)
    for k, (text, colour, kind) in enumerate(entries):
        line_key(x, y_mm + k * step, text, colour, dotted=kind == "dotted")


for key, (x0, x1) in COLUMNS.items():
    canvas.text((x0 + x1) / 2, 3.2, TITLES[key], fontsize=6, fontweight="bold", ha="center", va="center")

for row, case in enumerate(cases):
    top = ROW_Y[row]
    bottom_row = row == len(cases) - 1
    estimate = case["final"]["estimates"][0]["estimate"]
    fps, fs_gt = float(case["final"]["fps"]), float(case["final"]["fs_gt"])
    belt = filter_RW(recorded_belt(case["final"]["gt"]), fs_gt, lo=MIN_HZ, hi=MAX_HZ).reshape(-1)
    n_video = np.asarray(estimate["signal_hat"]).size
    video_time = np.arange(n_video) / fps
    belt_time = np.arange(belt.size) / fs_gt
    shown = belt_time <= video_time[-1]
    trial_span = (0.0, float(video_time[-1]))
    if case["name"] == "MAHNOB-HCI":
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
    canvas.text(0.6, top - HEADER + 1.6, "ab"[row], fontsize=8, fontweight="bold", va="center")
    # the panel names its dataset only; the caption identifies the trial and gives its error
    canvas.text(4.4, top - HEADER + 1.6, case["name"], fontsize=6, va="center")
    print(f"{name}; trial MAE {case['mae']:.2f} bpm")

    # observations: the belt and the three observations of the bank, one lane each
    x0, x1 = COLUMNS["observations"]
    traces = axes_mm(x0, top, x1, top + PANEL_H, span)
    traces.set_ylim(-3 * LANE_UNITS - LANE_UNITS / 2 - 0.25, LANE_UNITS / 2)
    z_belt = median_z(belt[shown])
    draw_trace(traces, belt_time[shown], z_belt, INK, gain=GAIN, lw=0.5)
    canvas.text(LABEL_X, top + PANEL_H * 0.5 / 4.15, "Belt", fontsize=5.5, ha="right", va="center")
    first = int(OBSERVATION_START_SEC * fps)
    for lane, (method, _, label, colour) in enumerate(OBSERVATIONS, start=1):
        observation = case["observations"][method][first:n_video]
        observation = filter_RW(observation, fps, lo=MIN_HZ, hi=MAX_HZ).reshape(-1) if observation.std() > 0 else observation
        time = video_time[first:first + observation.size]
        # each observation in its region's colour, as in Fig. 1c and in its update share
        draw_trace(traces, time, median_z(observation), colour, offset=-lane * LANE_UNITS, gain=GAIN, lw=0.45)
        canvas.text(LABEL_X, top + PANEL_H * (lane + 0.5) / 4.15, SHORT.get(label, label), fontsize=5.5,
                    ha="right", va="center")
    time_axis(traces, span, 10 if span[1] - span[0] <= 70 else 20, bottom_row)

    # update share of each observation in the Kalman update
    x0, x1 = COLUMNS["share"]
    precision = 1.0 / np.asarray(estimate["diagnostics"]["R_eff_t_channels"], dtype=float)
    update_share = precision / precision.sum(axis=0)
    families = json.loads(estimate["meta"])["observation_families"]
    share_time = np.linspace(0, video_time[-1], update_share.shape[1])
    for lane, (_, family, label, colour) in enumerate(OBSERVATIONS):
        y = top + 1.0 + lane * (SHARE_H + SHARE_GAP)
        share_axes = axes_mm(x0, y, x1, y + SHARE_H, span)
        draw_share(share_axes, share_time, update_share[families.index(family)], colour)
        canvas.text(x0 - 1.2, y + SHARE_H / 2, SHORT.get(label, label), fontsize=5.5, ha="right", va="center")
        if lane == len(OBSERVATIONS) - 1:
            time_axis(share_axes, span, 20, bottom_row)

    # output waveform against the belt, both scaled the same way, without lag or sign correction
    x0, x1 = COLUMNS["waveform"]
    waveform = axes_mm(x0, top, x1, top + PANEL_H, span)
    waveform.set_ylim(-3.4, 3.4)
    z_smoothed = median_z(np.asarray(estimate["z_full_smoothed"], dtype=float))
    draw_trace(waveform, belt_time[shown], z_belt, INK, lw=0.55)
    draw_trace(waveform, video_time[:z_smoothed.size], z_smoothed, PARH, lw=0.55)
    time_axis(waveform, span, 10 if span[1] - span[0] <= 70 else 20, bottom_row)

    # window rates over the whole trial, drawn at the window centres, so the axis runs from the first centre to the
    # last; the span of the other panels is shaded where it is shorter than the trial
    x0, x1 = COLUMNS["rate"]
    centre = case["windows"].window + WIN / 2
    centre_span = (float(centre.min()), float(centre.max()))
    rate = axes_mm(x0, top, x1, top + PANEL_H, centre_span)
    values = case["windows"][rate_columns]
    low = np.floor(min(values.min().min(), case["constant"])) - 1
    high = np.ceil(max(values.max().max(), case["constant"])) + 1
    if row == 0:  # room above the tracks for the key of this column
        high = max(high, np.ceil(low + (max(values.max().max(), case["constant"]) - low) / 0.62))
    rate.set_ylim(low, high)
    if span != trial_span:
        rate.axvspan(*span, color="#ececec", lw=0, zorder=0)
        rate.text((span[0] + span[1]) / 2, high - 0.03 * (high - low), "shown at left", fontsize=4.8, color="#555555",
                  ha="center", va="top")
    rate.plot([centre.min(), centre.max()], [case["constant"]] * 2, color=NEUTRAL, lw=0.8, ls=(0, (1, 1.2)), zorder=1)
    for key, colour in (("hilbert_reference_bpm", INK), ("ossm_kf_track_bpm", OSSM), ("parh_track_bpm", PARH)):
        rate.plot(centre, case["windows"][key], color=colour, lw=0.9, zorder=2)
    rate.spines["left"].set_visible(True)
    rate.spines["left"].set_linewidth(0.4)
    rate.yaxis.set_major_locator(MultipleLocator(2 if high - low <= 10 else 4))
    rate.tick_params(axis="y", length=1.5, pad=1, labelsize=5)
    time_axis(rate, centre_span, 10 if centre_span[1] - centre_span[0] <= 40 else 20, bottom_row)

# each column's key at its upper right in the first row, where the marks it names are first read
y_key = ROW_Y[0] - HEADER + 1.6
keys_in_line(COLUMNS["share"][1], y_key, [("Chest ROI", CHEST, "area"), ("Torso box", TORSO, "area")])
keys_in_line(COLUMNS["waveform"][1], y_key, [("Belt", INK, "line"), ("PARH-OSSM", PARH, "line")])
keys_stacked(COLUMNS["rate"][1], ROW_Y[0] + 1.4, [("Hilbert reference", INK, "line"), ("PARH-OSSM", PARH, "line"),
                                                 ("OSSM-KF", OSSM, "line"), ("Constant predictor", NEUTRAL, "dotted")])

fig.savefig(out_path)
fig.savefig(out_path.removesuffix(".pdf") + ".png", dpi=300)
print(f"height {HEIGHT:.1f} mm")
