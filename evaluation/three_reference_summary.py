"""Summarise window_rates_three_references.py: every method against each reference definition.

Per trial, the MAE of a method against a reference is the mean |estimate - reference| over the
windows where both exist; a trial enters a reference's comparison only with at least 10 such
windows. The constant predictor outputs, for each reference, the median over trials of that
reference's trial mean. Paired comparisons use the subject as the unit (median of the trial
differences per subject, Wilcoxon over subjects), as the paper's subject-level statistics do.

A per-window median-fusion CSV (median_fusion_window_rates.py), when given, adds that method.

usage: three_reference_summary.py <dataset csv> <MAHNOB root or "-"> [exclude sessions file or "-"] [median fusion csv]
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REFERENCES = ["hilbert", "welch", "count"]
METHODS = ["parh_track", "ossm_kf_track", "direct_p1d_welch", "constant"]
MIN_WINDOWS = 10

windows = pd.read_csv(sys.argv[1])
if sys.argv[2] != "-":
    root = sys.argv[2]
    subject_of = {f"mahnob_{s}": int(re.match(r"Part_(\d+)_", b).group(1))
                  for s in os.listdir(root) for b in os.listdir(os.path.join(root, s)) if b.lower().endswith(".bdf")}
else:
    subject_of = {t: int(t.split("_")[1]) for t in windows.trial.unique()}
if len(sys.argv) > 4:
    fusion = pd.read_csv(sys.argv[4])
    windows = windows.merge(fusion, on=["trial", "window"], how="left")
    METHODS.insert(3, "median_fusion")
if len(sys.argv) > 3 and sys.argv[3] != "-":
    excluded = {f"mahnob_{s.strip()}" for s in open(sys.argv[3]) if s.strip()}
    windows = windows[~windows.trial.isin(excluded)]
windows = windows[windows.trial.isin(windows.dropna(subset=["direct_p1d_welch_bpm"]).trial.unique())]

trial_means = windows.groupby("trial")[[f"{r}_reference_bpm" for r in REFERENCES]].mean()
cohort = {r: trial_means[f"{r}_reference_bpm"].median() for r in REFERENCES}
for r in REFERENCES:
    windows[f"constant_{r}"] = cohort[r]

rows = []
for trial, w in windows.groupby("trial"):
    for r in REFERENCES:
        reference = w[f"{r}_reference_bpm"]
        for m in METHODS:
            estimate = w[f"constant_{r}"] if m == "constant" else w[f"{m}_bpm"]
            valid = reference.notna() & estimate.notna()
            if valid.sum() >= MIN_WINDOWS:
                rows.append((trial, subject_of[trial], r, m, np.abs(estimate[valid] - reference[valid]).mean()))
mae = pd.DataFrame(rows, columns=["trial", "subject", "reference", "method", "MAE"])
print(f"{windows.trial.nunique()} trials, {len(set(subject_of[t] for t in windows.trial.unique()))} subjects; "
      f"cohort rate " + ", ".join(f"{r} {cohort[r]:.2f}" for r in REFERENCES) + " bpm")

print("\nReference disagreement, per trial mean |difference| over windows (median over trials, bpm):")
for a, b in [("hilbert", "welch"), ("hilbert", "count"), ("welch", "count")]:
    d = windows.assign(gap=(windows[f"{a}_reference_bpm"] - windows[f"{b}_reference_bpm"]).abs()).groupby("trial").gap.mean()
    print(f"  {a} vs {b}: median {d.median():.2f}, p90 {d.quantile(.9):.2f}")
print(f"  count reference missing in {windows.count_reference_bpm.isna().mean():.1%} of windows")

print("\nMedian MAE (bpm) / share of trials with MAE > 5 bpm / p90, by reference:")
table = mae.groupby(["method", "reference"]).MAE.agg(
    median="median", over_5=lambda x: np.mean(x > 5), p90=lambda x: x.quantile(.9), trials="size").unstack("reference")
print(table.round(3).reindex(METHODS).to_string())

print("\nPARH against each method, subject level (median of subject medians of the trial differences, "
      "subjects favouring PARH, Wilcoxon p):")
for r in REFERENCES:
    pivot = mae[mae.reference == r].pivot_table(index=["trial", "subject"], columns="method", values="MAE").dropna()
    for m in [m for m in METHODS if m != "parh_track"]:
        per_subject = (pivot[m] - pivot["parh_track"]).groupby(level="subject").median()
        trial_diff = pivot[m] - pivot["parh_track"]
        print(f"  {r:7s} {m:17s}: trials {len(pivot)}, trial median {trial_diff.median():+.3f} "
              f"(PARH better {np.mean(trial_diff > 0):.0%}), subject median {per_subject.median():+.3f}, "
              f"{int((per_subject > 0).sum())}/{per_subject.size}, p {wilcoxon(per_subject).pvalue:.2g}")
