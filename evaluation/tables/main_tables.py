"""Rows of the manuscript's respiratory-rate table (Table 2) for one dataset, as LaTeX.

Scored as three_reference_summary.py and confirmatory_test.py score: per trial and reference, the MAE is the
mean |estimate - reference| over the windows where both exist, with at least 10 such windows, on the trials
where direct P1D is computable. Each row gives the median trial MAE under the Hilbert, Welch, and breath-count
references, then, under the Hilbert reference, which the figures and in-text values use: the median trial RMSE,
the 90th percentile of the trial MAE over the scored trials (the large errors),
the median within-trial Pearson correlation between the window estimates and the window references (tracking
within a trial), the least-squares slope of the trial-mean estimate on the trial-mean reference over the scored
trials (1 when the estimate follows the differences between trials, 0 when it outputs one rate for all of them),
and the number of trials scored. The constant predictor, the last row, outputs the median trial-mean
reference of the development trials (all trials for COHFACE, whose every trial is development data), so on
MAHNOB-HCI test subjects it knows the population rate from other subjects only; it has no correlation, and
its slope is 0 by construction.

usage: main_tables.py <dataset label> <split csv | none> <evaluated split: test | development | all>
                      "<method label>|<rate source>=<window rates csv>:<column>" ...
  The first method's csv defines the windows, the references, and the common trials.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import linregress

REFERENCES = ["hilbert", "welch", "count"]
MIN_WINDOWS = 10

dataset, split_path, evaluated_split, *method_args = sys.argv[1:]
sources = []
for argument in method_args:
    names, source = argument.split("=", 1)
    path, column = source.rsplit(":", 1)
    sources.append((*names.split("|"), path, column))
methods = [label for label, _, _, _ in sources] + ["Constant predictor"]
rate_source = {label: kind for label, kind, _, _ in sources} | {"Constant predictor": "development median"}

first = pd.read_csv(sources[0][2])
first = first[first.trial.isin(first.dropna(subset=["direct_p1d_welch_bpm"]).trial.unique())]
windows = first[["trial", "window", *(f"{r}_reference_bpm" for r in REFERENCES)]]
for label, _, path, column in sources:
    rates = pd.read_csv(path)[["trial", "window", column]].rename(columns={column: label})
    windows = windows.merge(rates, on=["trial", "window"], how="left")

if split_path == "none":
    windows["subject"] = windows.trial.str.split("_").str[1]
    windows["split"] = "development"
else:
    split = pd.read_csv(split_path).set_index("trial")
    windows["subject"] = windows.trial.map(split.subject)
    windows["split"] = windows.trial.map(split.split)
development_means = windows[windows.split == "development"].groupby("trial")[
    [f"{r}_reference_bpm" for r in REFERENCES]].mean()
for r in REFERENCES:
    windows[f"Constant predictor_{r}"] = development_means[f"{r}_reference_bpm"].median()
if evaluated_split != "all":
    windows = windows[windows.split == evaluated_split]

rows = []
for trial, w in windows.groupby("trial"):
    for r in REFERENCES:
        reference = w[f"{r}_reference_bpm"]
        for method in methods:
            estimate = w[f"{method}_{r}" if method == "Constant predictor" else method]
            valid = reference.notna() & estimate.notna()
            if valid.sum() < MIN_WINDOWS:
                continue
            error = estimate[valid] - reference[valid]
            correlation = np.nan if estimate[valid].nunique() == 1 else np.corrcoef(estimate[valid], reference[valid])[0, 1]
            rows.append((trial, r, method, np.abs(error).mean(), np.sqrt(np.mean(error ** 2)), correlation,
                         estimate[valid].mean(), reference[valid].mean()))
scores = pd.DataFrame(rows, columns=["trial", "reference", "method", "MAE", "RMSE", "r", "mean_estimate", "mean_reference"])

medians = scores.groupby(["method", "reference"])[["MAE", "RMSE", "r"]].median()
hilbert = scores[scores.reference == "hilbert"]
trials = hilbert.groupby("method").trial.nunique()
slope = {method: linregress(t.mean_reference, t.mean_estimate).slope for method, t in hilbert.groupby("method")}
large_error = hilbert.groupby("method").MAE.quantile(0.9)
print(f"% {dataset}: {windows.trial.nunique()} trials, {windows.subject.nunique()} subjects "
      f"(split {evaluated_split}); breath-count reference in {scores[scores.reference == 'count'].trial.nunique()} trials")
for method in methods:
    mae = " & ".join(f"{medians.loc[(method, r), 'MAE']:.2f}" for r in REFERENCES)
    rmse = medians.loc[(method, "hilbert"), "RMSE"]
    correlation = medians.loc[(method, "hilbert"), "r"]
    correlation = "--" if np.isnan(correlation) else f"{correlation:.2f}"
    print(f"{method} & {rate_source[method]} & {mae} & {rmse:.2f} & {large_error[method]:.2f} & {correlation} & "
          f"{slope[method]:.2f} & {trials[method]} \\\\")
