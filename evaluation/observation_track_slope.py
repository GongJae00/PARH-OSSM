"""Do the nine observations, before PARH-OSSM combines them, follow the differences in rate between trials?

Per MAHNOB-HCI trial, each observation's OSSM-KF rate track (the `__kfstd` estimate of the per-observation pickle,
the input of the readout) is averaged over the trial, and the median over the nine observations is taken. Its
least-squares slope on the trial-mean Hilbert reference (the windows of the frozen onset scoring, at least 10 per
trial) is 1 when the observations follow the trials and 0 when they sit at one rate; its median says where they sit.
The same is printed for the frozen PARH-OSSM track, so the two can be set side by side.

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
    observation_means = [60 * np.nanmean(e["estimate"]["track_hz"]) for e in observations if e["method"].endswith("__kfstd")]
    parh_mean = 60 * np.nanmean([e["estimate"]["track_hz"] for e in final if e["method"] == "parh_ossm"][0])
    return trial, np.median(observation_means), parh_mean, len(observation_means)


split = pd.read_csv(os.path.join(records_dir, "mahnob_subject_split.csv"))
windows = pd.read_csv(os.path.join(records_dir, "mahnob_frozen_onset_window_rates_three_references.csv"))
windows = windows[windows.trial.isin(split[split.split == evaluated_split].trial)]
windows = windows[windows.trial.isin(windows.dropna(subset=["direct_p1d_welch_bpm"]).trial.unique())]
reference = windows.dropna(subset=["hilbert_reference_bpm"]).groupby("trial").filter(lambda w: len(w) >= MIN_WINDOWS)
reference = reference.groupby("trial").hilbert_reference_bpm.mean()

with Pool(workers) as pool:
    means = pd.DataFrame(pool.map(trial_means, reference.index), columns=["trial", "observations", "parh", "count"]).set_index("trial")
print(f"MAHNOB-HCI {evaluated_split}: {len(means)} trials, {means['count'].min()}-{means['count'].max()} observation tracks per trial; "
      f"trial-mean reference median {reference.median():.2f}, SD {reference.std():.2f} bpm")
for column, name in [("observations", "median of the observations' OSSM-KF tracks"), ("parh", "PARH-OSSM track")]:
    fit = linregress(reference, means[column])
    print(f"  {name:44s} slope {fit.slope:+.2f}  r {fit.rvalue:+.2f}  median {means[column].median():.2f}  SD {means[column].std():.2f} bpm")
