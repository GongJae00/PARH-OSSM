"""Agreement of the trial-mean rate with the reference (scatter, Bland-Altman, rate terciles).

Every method is scored against each of the three reference definitions on the same windows
(window_rates_three_references.py, plus median_fusion_window_rates.py). Per trial and reference,
the estimate and the reference are averaged over the windows where both exist, with at least 10 such
windows; the per-trial MAE is the mean |estimate - reference| over those windows. The constant
predictor outputs, in every window, the median over trials of the trial-mean reference rate, as in
three_reference_summary.py. Trials enter only when direct P1D is computable, the common set of the
paper's paired analyses.

Agreement is between trial means: the least-squares slope and Pearson r of the estimate on the
reference, and the Bland-Altman bias (estimate minus reference) with limits of agreement at
bias +/- 1.96 SD. Terciles split the trials by their trial-mean reference rate, per reference.

usage: rate_agreement.py <dataset name> <window-rate csv> <median fusion csv>:<column> <trial-means csv out>
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

REFERENCES = ["hilbert", "welch", "count"]
METHODS = ["direct_p1d_welch", "median_fusion", "ossm_kf_track", "parh_track", "constant"]
MIN_WINDOWS = 10

dataset = sys.argv[1]
fusion_csv, fusion_column = sys.argv[3].rsplit(":", 1)
fusion = pd.read_csv(fusion_csv)[["trial", "window", fusion_column]].rename(columns={fusion_column: "median_fusion_bpm"})
windows = pd.read_csv(sys.argv[2]).merge(fusion, on=["trial", "window"], how="left")
windows = windows[windows.trial.isin(windows.dropna(subset=["direct_p1d_welch_bpm"]).trial.unique())]

trial_reference = windows.groupby("trial")[[f"{r}_reference_bpm" for r in REFERENCES]].mean()
rows = []
for r in REFERENCES:
    cohort_rate = trial_reference[f"{r}_reference_bpm"].median()
    for trial, w in windows.groupby("trial"):
        reference = w[f"{r}_reference_bpm"]
        for m in METHODS:
            estimate = pd.Series(cohort_rate, index=w.index) if m == "constant" else w[f"{m}_bpm"]
            valid = reference.notna() & estimate.notna()
            if valid.sum() >= MIN_WINDOWS:
                rows.append((trial, r, m, estimate[valid].mean(), reference[valid].mean(),
                             np.abs(estimate[valid] - reference[valid]).mean()))
trials = pd.DataFrame(rows, columns=["trial", "reference", "method", "estimate_bpm", "reference_bpm", "MAE"])
trials["difference_bpm"] = trials.estimate_bpm - trials.reference_bpm
trials.to_csv(sys.argv[4], index=False, float_format="%.4f")

print(f"{dataset}: {windows.trial.nunique()} trials with direct P1D")
print("\nAgreement of trial means with the reference (slope, Pearson r, Bland-Altman bias and limits, bpm):")
for (r, m), g in trials.groupby(["reference", "method"], sort=False):
    bias, spread = g.difference_bpm.mean(), 1.96 * g.difference_bpm.std(ddof=1)
    if m == "constant":
        slope, r_value = 0.0, float("nan")
    else:
        fit = stats.linregress(g.reference_bpm, g.estimate_bpm)
        slope, r_value = fit.slope, fit.rvalue
    print(f"  {r:7s} {m:17s}: trials {len(g)}, reference mean {g.reference_bpm.mean():5.2f} SD {g.reference_bpm.std(ddof=1):4.2f}; "
          f"slope {slope:+.3f}, r {r_value:+.3f}; bias {bias:+.2f}, limits [{bias - spread:+.2f}, {bias + spread:+.2f}]")

print("\nBy tercile of the trial-mean reference rate: median MAE / mean bias / share of trials with MAE > 5 bpm")
for r in REFERENCES:
    g = trials[trials.reference == r].copy()
    edges = g[g.method == "parh_track"].reference_bpm.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
    g["tercile"] = pd.cut(g.reference_bpm, edges, labels=["low", "middle", "high"], include_lowest=True)
    print(f"  {r}: tercile edges {', '.join(f'{e:.1f}' for e in edges)} bpm")
    table = g.groupby(["method", "tercile"], observed=True).agg(
        median_MAE=("MAE", "median"), bias=("difference_bpm", "mean"), over_5=("MAE", lambda x: np.mean(x > 5)))
    print(table.unstack("tercile").reindex(METHODS).round(2).to_string())
