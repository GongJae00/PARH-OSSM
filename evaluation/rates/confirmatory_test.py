"""The confirmatory comparison on the MAHNOB-HCI test subjects.

Six subject-level comparisons form one Benjamini-Hochberg family: the frozen method against the
constant predictor and against the submitted PARH, under each of the three reference definitions.
Per trial and reference, the MAE is the mean |estimate - reference| over the windows where both
exist, with at least 10 such windows, on the trials where direct P1D is computable (the common set,
as in three_reference_summary.py). Per subject, the gain is the median over its trials of
(comparator MAE - frozen MAE), positive when the frozen method is better, and the test is the
Wilcoxon signed-rank test over subjects.

The constant predictor outputs the median trial-mean reference rate of the development trials, so
it knows the population rate from other subjects only. The same constant fitted on the evaluated
trials themselves, the best any constant can do on them, is reported beside it and is not tested.

Descriptive, with 95% intervals from 2000 bootstrap resamples of subjects: the least-squares slope
of the trial-mean estimate on the trial-mean reference (1 when the estimate follows the reference
between trials, 0 when it stays at one rate); the median over subjects of the within-subject
Spearman correlation between the trial means (does the estimate follow a subject's rate from one
trial to the next); and the median over trials of the Pearson correlation between the window
estimates and the window references (tracking within a trial, the r of the paper's Table 2).

usage: confirmatory_test.py <frozen window rates csv> <submitted window rates csv> <split csv> <evaluated split: test|development>
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, linregress, spearmanr, wilcoxon

REFERENCES = ["hilbert", "welch", "count"]
MIN_WINDOWS = 10
BOOTSTRAP = 2000

frozen = pd.read_csv(sys.argv[1])
submitted = pd.read_csv(sys.argv[2])
split = pd.read_csv(sys.argv[3]).set_index("trial")
evaluated_split = sys.argv[4]

paired = frozen.merge(submitted, on=["trial", "window"], suffixes=("", "_submitted"))
same_reference = all(np.allclose(paired[f"{r}_reference_bpm"], paired[f"{r}_reference_bpm_submitted"], equal_nan=True)
                     for r in REFERENCES)
print(f"same reference windows in the two files: {same_reference}")
if not same_reference:
    sys.exit(1)
windows = paired[paired.trial.isin(paired.dropna(subset=["direct_p1d_welch_bpm"]).trial.unique())].copy()
windows["subject"] = windows.trial.map(split.subject)
windows["split"] = windows.trial.map(split.split)

trial_reference = windows.groupby(["split", "trial"])[[f"{r}_reference_bpm" for r in REFERENCES]].mean()
constant_development = {r: trial_reference.loc["development", f"{r}_reference_bpm"].median() for r in REFERENCES}
constant_evaluated = {r: trial_reference.loc[evaluated_split, f"{r}_reference_bpm"].median() for r in REFERENCES}
windows = windows[windows.split == evaluated_split]
print(f"{windows.trial.nunique()} {evaluated_split} trials, {windows.subject.nunique()} subjects; constant from development "
      + ", ".join(f"{r} {constant_development[r]:.2f}" for r in REFERENCES)
      + "; fitted on the evaluated trials " + ", ".join(f"{r} {constant_evaluated[r]:.2f}" for r in REFERENCES) + " bpm")

rows = []
for (trial, subject), w in windows.groupby(["trial", "subject"]):
    for r in REFERENCES:
        reference = w[f"{r}_reference_bpm"]
        estimates = {"frozen": w.parh_track_bpm, "submitted": w.parh_track_bpm_submitted,
                     "constant": pd.Series(constant_development[r], index=w.index),
                     "constant_fitted": pd.Series(constant_evaluated[r], index=w.index)}
        for method, estimate in estimates.items():
            valid = reference.notna() & estimate.notna()
            if valid.sum() >= MIN_WINDOWS:
                rows.append((trial, subject, r, method, np.abs(estimate[valid] - reference[valid]).mean(),
                             estimate[valid].mean(), reference[valid].mean()))
mae = pd.DataFrame(rows, columns=["trial", "subject", "reference", "method", "MAE", "estimate_mean", "reference_mean"])

print("\nMedian trial MAE (bpm):")
print(mae.pivot_table(index="method", columns="reference", values="MAE", aggfunc="median").round(3).to_string())

tests = []
for r in REFERENCES:
    pivot = mae[mae.reference == r].pivot_table(index=["trial", "subject"], columns="method", values="MAE").dropna()
    for comparator in ["constant", "submitted", "constant_fitted"]:
        gain = (pivot[comparator] - pivot.frozen).groupby(level="subject").median()
        tests.append({"reference": r, "comparator": comparator, "trials": len(pivot), "subject median gain": gain.median(),
                      "subjects better": f"{int((gain > 0).sum())}/{gain.size}",
                      "worse by > 0.5": int((gain < -0.5).sum()), "p": wilcoxon(gain).pvalue})
tests = pd.DataFrame(tests)
confirmatory = tests.comparator != "constant_fitted"
tests.loc[confirmatory, "q (BH, 6 tests)"] = false_discovery_control(tests.loc[confirmatory, "p"])
print("\nFrozen method against each comparator (gain = comparator MAE - frozen MAE; constant_fitted is not tested):")
print(tests.round(4).to_string(index=False))

within_trial = []
for (trial, subject), w in windows.groupby(["trial", "subject"]):
    for r in REFERENCES:
        for method, column in [("frozen", "parh_track_bpm"), ("submitted", "parh_track_bpm_submitted")]:
            valid = w[f"{r}_reference_bpm"].notna() & w[column].notna()
            if valid.sum() >= MIN_WINDOWS and w.loc[valid, column].std() > 0 and w.loc[valid, f"{r}_reference_bpm"].std() > 0:
                within_trial.append((trial, subject, r, method, np.corrcoef(w.loc[valid, column], w.loc[valid, f"{r}_reference_bpm"])[0, 1]))
within_trial = pd.DataFrame(within_trial, columns=["trial", "subject", "reference", "method", "r"])

rng = np.random.default_rng(0)
print("\nTrial-mean agreement and within-trial tracking (95% subject-bootstrap interval):")
for r in REFERENCES:
    for method in ["frozen", "submitted"]:
        means = mae[(mae.reference == r) & (mae.method == method)]
        by_subject = {s: g for s, g in means.groupby("subject")}
        r_by_subject = {s: g.r for s, g in within_trial[(within_trial.reference == r) & (within_trial.method == method)].groupby("subject")}
        subjects = np.array(sorted(by_subject))

        def descriptives(chosen):
            sample = pd.concat([by_subject[s] for s in chosen])
            tracking = [spearmanr(by_subject[s].estimate_mean, by_subject[s].reference_mean).statistic for s in chosen]
            window_r = np.concatenate([r_by_subject[s].to_numpy() for s in chosen if s in r_by_subject])
            return linregress(sample.reference_mean, sample.estimate_mean).slope, np.nanmedian(tracking), np.median(window_r)

        point = descriptives(subjects)
        resampled = np.array([descriptives(rng.choice(subjects, subjects.size)) for _ in range(BOOTSTRAP)])
        low, high = np.nanpercentile(resampled, [2.5, 97.5], axis=0)
        print(f"  {r:7s} {method:9s}: slope {point[0]:.3f} [{low[0]:.3f}, {high[0]:.3f}], "
              f"within-subject Spearman median {point[1]:.3f} [{low[1]:.3f}, {high[1]:.3f}], "
              f"within-trial r median {point[2]:.3f} [{low[2]:.3f}, {high[2]:.3f}]")
