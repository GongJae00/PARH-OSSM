"""Do the nine observations, before PARH-OSSM combines them, follow the differences in rate between trials?

Per MAHNOB-HCI trial, each observation's OSSM-KF rate track (the `__kfstd` estimate of the per-observation pickle,
the input of the readout) is averaged over the trial. The least-squares slope of these trial means on the trial-mean
Hilbert reference (the windows of the frozen onset scoring, at least 10 per trial) is 1 when an observation follows
the trials and 0 when it sits at one rate; the median of the trial means says where it sits. The same is printed for
each observation, for the median over the nine, and for the frozen PARH-OSSM track, so they can be set side by side.

usage: observation_track_slope.py <records dir> <per-observation data dir> <frozen final data dir> <evaluated split: development|test> [workers]
<records dir> holds the CSVs of window_rates.sh and the MAHNOB-HCI subject split.
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import linregress

MIN_WINDOWS = 10
records_dir, observation_dir, final_dir, evaluated_split = sys.argv[1:5]
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 4


def trial_means(trial):
    observations = pickle.load(open(os.path.join(observation_dir, f"{trial}.pkl"), "rb"))["estimates"]
    final = pickle.load(open(os.path.join(final_dir, f"{trial}.pkl"), "rb"))["estimates"]
    observation_means = {e["method"].removesuffix("__kfstd"): 60 * np.nanmean(e["estimate"]["track_hz"])
                         for e in observations if e["method"].endswith("__kfstd")}
    observation_means["median of the observations' OSSM-KF tracks"] = np.median(list(observation_means.values()))
    observation_means["PARH-OSSM track"] = 60 * np.nanmean([e["estimate"]["track_hz"] for e in final if e["method"] == "parh_ossm"][0])
    return observation_means


split = pd.read_csv(os.path.join(records_dir, "mahnob_subject_split.csv"))
windows = pd.read_csv(os.path.join(records_dir, "mahnob_frozen_onset_window_rates_three_references.csv"))
windows = windows[windows.trial.isin(split[split.split == evaluated_split].trial)]
windows = windows[windows.trial.isin(windows.dropna(subset=["direct_p1d_welch_bpm"]).trial.unique())]
reference = windows.dropna(subset=["hilbert_reference_bpm"]).groupby("trial").filter(lambda w: len(w) >= MIN_WINDOWS)
reference = reference.groupby("trial").hilbert_reference_bpm.mean()

with Pool(workers) as pool:
    means = pd.DataFrame(pool.map(trial_means, reference.index), index=reference.index)
print(f"MAHNOB-HCI {evaluated_split}: {len(means)} trials; trial-mean reference median {reference.median():.2f}, SD {reference.std():.2f} bpm")
for name in means.columns:
    present = means[name].notna()  # an observation can be missing from a trial
    fit = linregress(reference[present], means[name][present])
    print(f"  {name:44s} trials {present.sum():3d}  slope {fit.slope:+.2f}  r {fit.rvalue:+.2f}  "
          f"median {means[name].median():.2f}  SD {means[name].std():.2f} bpm")
