"""Supplementary figure of trial-mean rate agreement, one reference definition per figure.

Rows: COHFACE estimate against reference, COHFACE Bland-Altman, then the same for MAHNOB-HCI.
Columns: the four video methods. The values come from rate_agreement.py's trial-means CSVs; MAHNOB-HCI is
restricted to the test participants, so its slope, r, bias and limits are computed on those trials. The constant predictor,
which outputs the cohort rate in every window, appears as a horizontal line in the agreement panels.

usage: rate_agreement_figure.py <records dir> <reference: hilbert|welch|count> <out.pdf> [MAHNOB-HCI split, default test]
<records dir> holds rate_agreement/, the trial-means CSVs written by evaluation/rate_agreement.py, and the
MAHNOB-HCI subject split (evaluation/mahnob_subject_split.csv).
"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import GUIDE, INK, MM, panel_letter

matplotlib.use("pdf")
POINT = "#4a4a4a"
ON_POINTS = [patheffects.withStroke(linewidth=1.5, foreground="white")]
METHODS = [("direct_p1d_welch", "Direct P1D quadratic"), ("median_fusion", "Median fusion"),
           ("ossm_kf_track", "OSSM-KF"), ("parh_track", "PARH-OSSM")]
DATASETS = [("COHFACE", "cohface_frozen_trial_means.csv"), ("MAHNOB-HCI", "mahnob_frozen_onset_trial_means.csv")]
REFERENCE_NAMES = {"hilbert": "Hilbert", "welch": "Welch", "count": "breath-count"}

records_dir, reference, out_path = sys.argv[1:4]
# MAHNOB-HCI on the evaluated split, the test participants for the main Tables 2 and 3
split = pd.read_csv(os.path.join(records_dir, "mahnob_subject_split.csv"))
mahnob_test_trials = set(split[split.split == (sys.argv[4] if len(sys.argv) > 4 else "test")].trial)
fig, axes = plt.subplots(4, 4, figsize=(183 * MM, 168 * MM))

for d, (dataset, file_name) in enumerate(DATASETS):
    trials = pd.read_csv(os.path.join(records_dir, "rate_agreement", file_name))
    trials = trials[trials.reference == reference]
    if dataset == "MAHNOB-HCI":
        trials = trials[trials.trial.isin(mahnob_test_trials)]
    cohort_rate = trials[trials.method == "constant"].estimate_bpm.iloc[0]
    shown = trials[trials.method != "constant"]
    low, high = np.floor(shown[["estimate_bpm", "reference_bpm"]].min().min()), np.ceil(shown[["estimate_bpm", "reference_bpm"]].max().max())
    mean_low, mean_high = np.floor(((shown.estimate_bpm + shown.reference_bpm) / 2).min()), np.ceil(((shown.estimate_bpm + shown.reference_bpm) / 2).max())
    difference_limit = np.ceil(shown.difference_bpm.abs().max())

    for c, (method, title) in enumerate(METHODS):
        g = trials[trials.method == method]
        agreement, bland_altman = axes[2 * d, c], axes[2 * d + 1, c]

        fit = stats.linregress(g.reference_bpm, g.estimate_bpm)
        agreement.plot([low, high], [low, high], color=GUIDE, lw=0.6, ls=(0, (3, 2)), zorder=1)
        agreement.axhline(cohort_rate, color=GUIDE, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        agreement.scatter(g.reference_bpm, g.estimate_bpm, s=3, color=POINT, alpha=0.35, lw=0, zorder=2)
        agreement.plot([low, high], [fit.intercept + fit.slope * low, fit.intercept + fit.slope * high], color=INK, lw=0.8, zorder=3)
        agreement.set(xlim=(low, high), ylim=(low, high), aspect="equal")
        agreement.set_title(f"slope {fit.slope:.2f}, r {fit.rvalue:.2f}".replace("-", "\u2212"), fontsize=5.5, pad=2)

        mean_rate = (g.estimate_bpm + g.reference_bpm) / 2
        bias, spread = g.difference_bpm.mean(), 1.96 * g.difference_bpm.std(ddof=1)
        bland_altman.axhline(0, color=GUIDE, lw=0.6, ls=(0, (3, 2)), zorder=1)
        bland_altman.scatter(mean_rate, g.difference_bpm, s=3, color=POINT, alpha=0.35, lw=0, zorder=2)
        bland_altman.axhline(bias, color=INK, lw=0.8, zorder=3)
        for limit in (bias - spread, bias + spread):
            bland_altman.axhline(limit, color=INK, lw=0.5, ls=(0, (4, 2)), zorder=3)
        bland_altman.set(xlim=(mean_low, mean_high), ylim=(-difference_limit, difference_limit))
        bland_altman.set_title(f"bias {bias:+.2f}, limits {bias - spread:+.2f} to {bias + spread:+.2f}".replace("-", "\u2212"),
                               fontsize=5.5, pad=2)

        if c == 0:
            # the lines are named once, where they separate: the MAHNOB-HCI agreement panel and the COHFACE Bland-Altman panel
            label = dict(fontsize=5, textcoords="offset points", path_effects=ON_POINTS, zorder=4)
            if dataset == "MAHNOB-HCI":
                corner = low + 0.82 * (high - low)
                agreement.annotate("identity", (corner, corner), xytext=(-2, 2), ha="right", va="bottom", color=GUIDE, **label)
                agreement.annotate("least-squares fit", (high, fit.intercept + fit.slope * high), xytext=(0, -2),
                                   ha="right", va="top", color=INK, **label)
                agreement.annotate("cohort rate", (low, cohort_rate), xytext=(2, -2), ha="left", va="top", color=GUIDE, **label)
            else:
                for level, name, offset, align in [(bias, "bias", 1.5, "bottom"), (bias + spread, "bias + 1.96 SD", 1.5, "bottom"),
                                                   (bias - spread, "bias \u2212 1.96 SD", -1.5, "top")]:
                    bland_altman.annotate(name, (mean_low, level), xytext=(2, offset), ha="left", va=align, color=INK, **label)
        if d == 0:
            agreement.text(0.5, 1.16, title, transform=agreement.transAxes, ha="center", fontsize=6.5)
        agreement.set_xlabel("Reference rate (bpm)")
        bland_altman.set_xlabel("Mean rate (bpm)")
        if c == 0:
            agreement.set_ylabel(f"{dataset}\nEstimated rate (bpm)")
            bland_altman.set_ylabel(f"{dataset}\nDifference (bpm)")

for row, label in enumerate("abcd"):
    panel_letter(axes[row, 0], label, -0.40, 1.08)
fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.05, wspace=0.38, hspace=0.55)
fig.savefig(out_path)
print(f"{out_path}: {REFERENCE_NAMES[reference]} reference")
