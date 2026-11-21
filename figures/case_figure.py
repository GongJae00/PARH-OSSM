"""Figure 2: one trial per dataset followed through the stages of PARH-OSSM, against the belt.

Columns: a COHFACE trial and a MAHNOB-HCI development trial, each the trial whose PARH-OSSM MAE against the
Hilbert reference (scored as main_tables.py scores it) is closest to the median over that dataset's
development trials, so neither case is chosen for looking good and no test subject is read. Rows:
- the five observations of the bank and the belt, each band-passed to 0.08-0.5 Hz as the evaluation filters the belt, centred
  on its median and scaled by its median absolute deviation, so that the filter's start-up and the end of a belt
  record do not set the scale; for display, values beyond about 2 are compressed smoothly toward 3
  (3 tanh(z / 3)), so that a motion burst does not hide the neighbouring traces; the samples more than 4
  deviations from the median, where the compression flattens the trace, are drawn in a lighter tone. The observations are filtered
  from 2 s on: their first frames carry the initialisation step of the extraction (in COHFACE up to 150 times the
  later spread), which the filter would spread over several seconds, and the observation calibration of the method
  skips the same 2 s (OBS_CAL_SKIP_SEC in parh_ossm.py);
- the trust pi of each observation in the filter (diagnostics pi_t_channels), on one colour scale for both that
  spans every value of the two trials;
- the output waveform that the waveform metrics score (z_full_smoothed, as bounded_lag_waveform.py reads it) and
  the belt, scaled the same way, without any lag or sign correction;
- the window rates that the tables score: the Hilbert reference, PARH-OSSM, the single-observation OSSM-KF,
  and the constant predictor, plotted at the window centres on the same span of rate in both columns.

usage: case_figure.py <COHFACE window rates csv> <COHFACE final dir> <COHFACE observation dir>
                      <MAHNOB-HCI window rates csv> <MAHNOB-HCI final dir> <MAHNOB-HCI observation dir>
                      <split csv> <out.pdf>
"""
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import offset_copy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.belt_reference import filter_RW, recorded_belt
from figure_style import INK, MM, NEUTRAL, OSSM, PARH

matplotlib.use("pdf")
WIN, MIN_WINDOWS, MIN_HZ, MAX_HZ = 30, 10, 0.08, 0.5
OBSERVATION_START_SEC = 2.0
OBSERVATIONS = [("of_farneback", "OF"), ("DoF", "DoF"), ("dof_disp_bridge", "DoF bridge"), ("profile1D quadratic", "P1D"),
                ("torso_shift", "Torso displacement")]
TRUST = LinearSegmentedColormap.from_list("trust", ["#f2f6fc", "#1d4f91"])
BEYOND = 4
LIGHTER = {INK: "#b4b4b4", NEUTRAL: "#d2d2d2", PARH: "#a8bfdf"}
TEXT_WIDTH_MM = 175.9


def plot_scaled(ax, time, x, color, offset=0.0, gain=1.0, lw=0.6, **kwargs):
    """Draw x centred on its median and scaled by its median absolute deviation, compressed as 3 tanh(z / 3); the
    samples beyond BEYOND deviations, where the compression flattens the trace, are drawn in a lighter tone."""
    spread = 1.4826 * np.median(np.abs(x - np.median(x)))
    z = (x - np.median(x)) / spread if spread > 0 else np.zeros_like(x)
    shown = 3 * np.tanh(z / 3) * gain + offset
    beyond = np.abs(z) > BEYOND
    joined = beyond | np.r_[beyond[1:], False] | np.r_[False, beyond[:-1]]
    ax.plot(time, np.where(beyond, np.nan, shown), color=color, lw=lw, **kwargs)
    ax.plot(time, np.where(joined, shown, np.nan), color=LIGHTER[color], lw=lw)


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

trust_values = np.concatenate([np.asarray(c["final"]["estimates"][0]["estimate"]["diagnostics"]["pi_t_channels"]).ravel()
                               for c in cases])
trust_range = (np.floor(trust_values.min() * 100) / 100, np.ceil(trust_values.max() * 100) / 100)
rate_columns = ["hilbert_reference_bpm", "parh_track_bpm", "ossm_kf_track_bpm"]
rate_limits = [(min(c["windows"][rate_columns].min().min(), c["constant"]),
                max(c["windows"][rate_columns].max().max(), c["constant"])) for c in cases]
rate_span = np.ceil(max(high - low for low, high in rate_limits)) + 1

fig = plt.figure(figsize=(TEXT_WIDTH_MM * MM, 126 * MM))
grid = fig.add_gridspec(4, 3, height_ratios=[1.75, 0.8, 0.9, 1.2], width_ratios=[1, 1, 0.025],
                        hspace=0.3, wspace=0.14, left=0.12, right=0.935, top=0.945, bottom=0.07)
for column, case in enumerate(cases):
    estimate = case["final"]["estimates"][0]["estimate"]
    fps, fs_gt = float(case["final"]["fps"]), float(case["final"]["fs_gt"])
    belt = filter_RW(recorded_belt(case["final"]["gt"]), fs_gt, lo=MIN_HZ, hi=MAX_HZ).reshape(-1)
    n_video = np.asarray(estimate["signal_hat"]).size
    video_time = np.arange(n_video) / fps
    belt_time = np.arange(belt.size) / fs_gt
    shown = belt_time <= video_time[-1]
    span = (0, video_time[-1])

    traces = fig.add_subplot(grid[0, column])
    plot_scaled(traces, belt_time[shown], belt[shown], INK, gain=0.4)
    start = int(OBSERVATION_START_SEC * fps)
    for row, (method, _) in enumerate(OBSERVATIONS, start=1):
        observation = case["observations"][method][start:n_video]
        observation = filter_RW(observation, fps, lo=MIN_HZ, hi=MAX_HZ).reshape(-1) if observation.std() > 0 else observation
        plot_scaled(traces, video_time[start:start + observation.size], observation, NEUTRAL, offset=-row * 1.6, gain=0.4,
                    lw=0.45)
    traces.set_yticks([-row * 1.6 for row in range(len(OBSERVATIONS) + 1)])
    traces.set_yticklabels(["Belt"] + [label for _, label in OBSERVATIONS] if column == 0 else [])
    traces.tick_params(axis="y", length=0)
    traces.spines["left"].set_visible(False)
    traces.set(xlim=span, ylim=(-len(OBSERVATIONS) * 1.6 - 1.3, 1.3), xticklabels=[])
    identifier = case["trial"].split("_")[1:]
    heading = offset_copy(traces.transAxes, fig=fig, y=6, units="points")
    traces.text(0, 1, f"{case['name']}, subject {identifier[0]}, trial {identifier[1]}" if case["name"] == "COHFACE"
                else f"{case['name']}, session {identifier[0]}", transform=heading, va="baseline")
    traces.text(-0.16 if column == 0 else -0.06, 1, "ab"[column], transform=heading, va="baseline", fontsize=8,
                fontweight="bold")

    trust = fig.add_subplot(grid[1, column])
    image = trust.imshow(np.asarray(estimate["diagnostics"]["pi_t_channels"]), aspect="auto", cmap=TRUST,
                         vmin=trust_range[0], vmax=trust_range[1], extent=(0, video_time[-1], len(OBSERVATIONS) - 0.5, -0.5),
                         interpolation="nearest")
    trust.set_yticks(range(len(OBSERVATIONS)))
    trust.set_yticklabels([label for _, label in OBSERVATIONS] if column == 0 else [])
    trust.tick_params(axis="y", length=0)
    trust.set(xlim=span, xticklabels=[])
    for side in ("left", "bottom"):
        trust.spines[side].set_visible(False)

    waveform = fig.add_subplot(grid[2, column])
    plot_scaled(waveform, belt_time[shown], belt[shown], INK, label="Belt")
    plot_scaled(waveform, video_time, np.asarray(estimate["z_full_smoothed"], dtype=float), PARH, label="PARH-OSSM waveform")
    waveform.set(xlim=span, xticklabels=[], ylim=(-3.3, 3.3))
    waveform.set_ylabel("Scaled signal" if column == 0 else "")

    rate = fig.add_subplot(grid[3, column])
    centre = case["windows"].window + WIN / 2
    rate.plot(centre, case["windows"].hilbert_reference_bpm, color=INK, lw=1.0, label="Hilbert reference")
    rate.plot(centre, case["windows"].parh_track_bpm, color=PARH, lw=1.0, label="PARH-OSSM")
    rate.plot(centre, case["windows"].ossm_kf_track_bpm, color=OSSM, lw=1.0, label="OSSM-KF")
    rate.axhline(case["constant"], color=NEUTRAL, lw=0.8, ls=(0, (1, 1.5)), label="Constant predictor")
    middle = sum(rate_limits[column]) / 2
    rate.set(xlim=span, ylim=(middle - rate_span / 2, middle + rate_span / 2), xlabel="Time (s)")
    rate.yaxis.set_major_locator(MaxNLocator(integer=True))
    rate.set_ylabel("Rate (bpm)" if column == 0 else "")
    if column == 0:
        waveform.legend(loc="lower left", bbox_to_anchor=(0, 0.98), fontsize=5, frameon=False, ncol=2,
                        handlelength=1.5, borderaxespad=0)
        rate.legend(loc="lower left", bbox_to_anchor=(0, 0.98), fontsize=5, frameon=False, ncol=4,
                    handlelength=1.0, columnspacing=0.8, borderaxespad=0)

colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[1, 2]))
colorbar.set_label("Trust $\\pi$")
colorbar.ax.tick_params(labelsize=5.5, length=1.5)
colorbar.set_ticks(np.arange(np.ceil(trust_range[0] * 20) / 20, trust_range[1] + 1e-9, 0.05))
colorbar.ax.yaxis.set_major_formatter("{x:.2f}")
fig.savefig(out_path)
fig.savefig(out_path.removesuffix(".pdf") + ".png", dpi=300)
