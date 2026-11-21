"""Respiratory-rate error by the amount of head motion measured in the video.

Motion is measured without the reference by head_motion.py: the face-detection box is tracked four
times a second, and the trial's motion is the 5-95th percentile range of the box centre (Euclidean
norm of the two image axes) in percent of the median face height. The mean speed of the 1-s smoothed
centre is a second measure that checks that the ranking does not depend on this choice. Both were
fixed, and checked against detection misses, detection jitter, and the frames of the extreme trials,
before any error was split by them.

Trials are split into motion tertiles within each dataset. Per trial, MAE against a reference is the
mean |estimate - reference| over windows where both exist, with at least 10 such windows, and the
constant predictor outputs the median trial-mean reference of the development trials, as in main_tables.py.
With a split csv (MAHNOB-HCI), the constant comes from the development trials and the test trials alone are
scored and split into tertiles; without one (COHFACE), every trial is development data and is scored. The
per-trial motion, tertile, and MAE go to a CSV for the Supplementary table.

usage: motion_tertiles.py <dataset name> <window-rate csv> <head_motion_trials.csv> <trial csv out> [split csv]
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REFERENCES = ["hilbert", "welch", "count"]
METHODS = ["parh_track", "ossm_kf_track", "direct_p1d_welch", "constant"]
MIN_WINDOWS = 10

dataset, window_csv, motion_csv, out_csv = sys.argv[1:5]
windows = pd.read_csv(window_csv)
motion = (pd.read_csv(motion_csv)
          .rename(columns={"range_pct_face": "motion_range", "speed_pct_face_per_sec": "speed"})
          .set_index("trial")[["motion_range", "speed", "miss_share"]])

windows = windows[windows.trial.isin(windows.dropna(subset=["direct_p1d_welch_bpm"]).trial.unique())]
trial_means = windows.groupby("trial")[[f"{r}_reference_bpm" for r in REFERENCES]].mean()
cohort_rate = trial_means.median()
if len(sys.argv) > 5:
    split = pd.read_csv(sys.argv[5])
    cohort_rate = trial_means[trial_means.index.isin(split[split.split == "development"].trial)].median()
    windows = windows[windows.trial.isin(split[split.split == "test"].trial)]
rows = []
for trial, w in windows.groupby("trial"):
    for r in REFERENCES:
        reference = w[f"{r}_reference_bpm"]
        for m in METHODS:
            estimate = (pd.Series(cohort_rate[f"{r}_reference_bpm"], index=w.index) if m == "constant"
                        else w[f"{m}_bpm"])
            valid = reference.notna() & estimate.notna()
            if valid.sum() >= MIN_WINDOWS:
                rows.append((trial, r, m, np.abs(estimate[valid] - reference[valid]).mean()))
mae = pd.DataFrame(rows, columns=["trial", "reference", "method", "MAE"])

motion = motion.loc[sorted(set(mae.trial))]
motion["tertile"] = pd.qcut(motion.motion_range, 3, labels=["low", "middle", "high"])
mae = mae.join(motion, on="trial")
mae.to_csv(out_csv, index=False, float_format="%.4f")

rho, p = spearmanr(motion.motion_range, motion.speed)
print(f"{dataset}: {len(motion)} trials; head-motion range vs speed: Spearman {rho:.2f} (p {p:.2g})")
for tertile, m in motion.groupby("tertile", observed=True):
    print(f"  {tertile:6s}: {len(m)} trials, range {m.motion_range.min():.1f}-{m.motion_range.max():.1f} % of face height "
          f"(median {m.motion_range.median():.1f}), detection missed in {m.miss_share.mean():.2%} of samples")

print("\nMedian trial MAE (bpm) by motion tertile:")
table = mae.groupby(["reference", "method", "tertile"], observed=True).MAE.median().unstack("tertile")
print(table.reindex(METHODS, level="method").round(2).to_string())

print("\nShare of trials with MAE > 5 bpm by motion tertile:")
table = mae.groupby(["reference", "method", "tertile"], observed=True).MAE.apply(lambda x: np.mean(x > 5)).unstack("tertile")
print(table.reindex(METHODS, level="method").round(3).to_string())

print("\nSpearman correlation of trial MAE with head-motion range (and with speed):")
for (r, m), g in mae.groupby(["reference", "method"]):
    rho, p = spearmanr(g.motion_range, g.MAE)
    rho_speed, p_speed = spearmanr(g.speed, g.MAE)
    print(f"  {r:7s} {m:17s}: rho {rho:+.2f} (p {p:.2g}); speed {rho_speed:+.2f} (p {p_speed:.2g}); {len(g)} trials")

print("\nShare of trials where PARH has a lower MAE than each method, by motion tertile:")
pivot = mae.pivot_table(index=["trial", "reference", "tertile"], columns="method", values="MAE", observed=True).dropna()
for m in [m for m in METHODS if m != "parh_track"]:
    share = (pivot[m] > pivot["parh_track"]).groupby(level=["reference", "tertile"], observed=True).mean().unstack("tertile")
    print(f"  vs {m}:\n" + share.round(2).to_string())
