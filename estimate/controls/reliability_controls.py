"""Control reliability priors: the frozen PARH with its readout reliability equalized or window-shuffled.

The readout (reliability_prior_scope readout_only) weights each observation family's rate candidate by the
GT-free prior of priors_group.csv, one row per trial, family and 30-s window (10-s stride). Two controls
remove one part of that information each while keeping everything else:

  equal     every family gets the across-family mean of each numeric column in each window, so the families'
            prior rows become identical: the readout can no longer prefer one observation over another, but
            the window-to-window level is kept. The counts in the row are averaged with the scores, and a
            score missing for one family takes the mean of the families that have it;
  shuffled  each family's windows are permuted within the trial, so each family keeps its own scores and
            its overall level but they no longer line up with the time they were computed for.

usage: reliability_controls.py <priors_group.csv> <equal|shuffled> <out csv> [seed]
"""
import sys

import numpy as np
import pandas as pd

WINDOW_COLUMNS = ["window_id", "window_start_sec", "window_end_sec", "window_duration_sec"]

priors_path, control, out_path = sys.argv[1:4]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
priors = pd.read_csv(priors_path)
scores = [c for c in priors.select_dtypes("number").columns if c not in WINDOW_COLUMNS]

if control == "equal":
    priors[scores] = priors.groupby(["video", "window_id"])[scores].transform("mean")
else:
    rng = np.random.default_rng(seed)
    for _, rows in priors.groupby(["video", "group"]).groups.items():
        priors.loc[rows, scores] = priors.loc[rng.permutation(rows), scores].to_numpy()

priors.to_csv(out_path, index=False)
