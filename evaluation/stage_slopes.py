"""How far each method and each stage of PARH-OSSM follows the differences in rate between trials.

Per method or stage, the trial-mean estimate is regressed on the trial-mean reference over the windows where both
exist (at least 10 per trial, on the trials with direct P1D, as in rate_agreement.py and main_tables.py).
A slope near 1 follows the between-trial differences of the reference; a slope near 0 does not, whether the trial means scatter (noise) or gather (a pull toward one rate). The
SD and median of the trial means tell the two apart and show where they gather. The code holds no population rate
(the reliability priors are per-observation scores, and the initial frequency comes from each video's first 10 s);
observation_track_slope.py shows whether the observations already gather before they are combined.

usage: stage_slopes.py <records dir> <reference: hilbert|welch|count> mahnob <evaluated split: development|test>
       stage_slopes.py <records dir> <reference: hilbert|welch|count> cohface
<records dir> holds the CSVs of window_rates.sh and bank_rates.sh and the MAHNOB-HCI subject split.
"""
import os
import sys

import pandas as pd
from scipy import stats

MIN_WINDOWS = 10
STAGES = ["plain_multichannel", "readout", "observation_law", "observability"]

records_dir, reference, dataset = sys.argv[1:4]
if dataset == "mahnob":
    split = pd.read_csv(os.path.join(records_dir, "mahnob_subject_split.csv"))
    trials = set(split[split.split == sys.argv[4]].trial)
    frozen = "mahnob_frozen_onset_window_rates_three_references.csv"
    stage_csv = "mahnob_{}_onset_window_rates_three_references.csv"
    fusion = "mahnob_observation_fusion_window_rates.csv"
else:
    trials = None
    frozen = "cohface_frozen_window_rates_three_references.csv"
    stage_csv = "cohface_{}_window_rates_three_references.csv"
    fusion = "cohface_observation_fusion_window_rates.csv"

windows = pd.read_csv(os.path.join(records_dir, frozen))
if trials is not None:
    windows = windows[windows.trial.isin(trials)]
# the trials where direct P1D is computable, the common set of Tables 2 and 3 (main_tables.py)
windows = windows[windows.trial.isin(windows.dropna(subset=["direct_p1d_welch_bpm"]).trial.unique())]
references = windows[["trial", "window", f"{reference}_reference_bpm"]]
fused = pd.read_csv(os.path.join(records_dir, fusion))
sources = [("Direct P1D quadratic", windows, "direct_p1d_welch_bpm"), ("Direct torso displacement", fused, "direct_torso_bpm"),
           ("Median fusion", fused, "median_fusion_nine_bpm"), ("OSSM-KF", windows, "ossm_kf_track_bpm")]
sources += [(stage, pd.read_csv(os.path.join(records_dir, stage_csv.format(stage))), "parh_track_bpm") for stage in STAGES]
sources += [("PARH-OSSM", windows, "parh_track_bpm")]
# PARH-OSSM with the torso observation left out of the bank (the leave-one-out run no_torso), the same flags and priors
sources += [("PARH-OSSM without torso", pd.read_csv(os.path.join(records_dir, stage_csv.format("loo9_no_torso"))), "parh_track_bpm")]

print(f"{dataset}, {reference} reference: trials, slope and Pearson r of trial-mean estimate on trial-mean reference, "
      f"SD and median of trial means (estimate / reference, bpm)")
for name, rates, column in sources:
    paired = references.merge(rates[["trial", "window", column]], on=["trial", "window"]).dropna()
    paired = paired.groupby("trial").filter(lambda w: len(w) >= MIN_WINDOWS).groupby("trial")[[f"{reference}_reference_bpm", column]].mean()
    fit = stats.linregress(paired[f"{reference}_reference_bpm"], paired[column])
    print(f"  {name:28s} {len(paired):4d}  slope {fit.slope:+.2f}  r {fit.rvalue:+.2f}  "
          f"SD {paired[column].std():.2f} / {paired[f'{reference}_reference_bpm'].std():.2f}  "
          f"median {paired[column].median():.2f} / {paired[f'{reference}_reference_bpm'].median():.2f}")
