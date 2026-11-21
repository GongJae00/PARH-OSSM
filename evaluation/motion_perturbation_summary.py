"""R1-4b: error of each method on the perturbed COHFACE copies against the unperturbed run (CHECKLIST, R1-4b).

One trial per subject (cohface_perturbation_trials.txt), so a trial stands for its subject. Scored as main_tables.py
scores: the trial MAE is the mean |estimate - reference| over the windows where both exist, with at least 10 such
windows. For each condition and reference the table gives, per method, the median trial MAE, the median of the
paired change against the unperturbed run (perturbed - unperturbed), and the number of subjects made worse by more
than 0.1 bpm. Descriptive only: development data, no test. The per-trial MAEs go to a csv for the Supplementary
table (supplementary_tables.py).

usage: motion_perturbation_summary.py <unperturbed window rates csv> <trials txt> <trial MAE csv out>
                                      <kind>:<level>=<csv> ... > <out.txt>
"""
import sys

import numpy as np
import pandas as pd

REFERENCES = ["hilbert", "welch", "count"]
METHODS = {"parh_track_bpm": "PARH-OSSM", "ossm_kf_track_bpm": "OSSM-KF", "direct_p1d_welch_bpm": "direct P1D"}
MIN_WINDOWS = 10
WORSE_BPM = 0.1


def trial_mae(path, trials):
    windows = pd.read_csv(path)
    windows = windows[windows.trial.isin(trials)]
    rows = []
    for trial, w in windows.groupby("trial"):
        for r in REFERENCES:
            for column, method in METHODS.items():
                valid = w[f"{r}_reference_bpm"].notna() & w[column].notna()
                if valid.sum() >= MIN_WINDOWS:
                    rows.append((trial, r, method, (w[column] - w[f"{r}_reference_bpm"])[valid].abs().mean()))
    return pd.DataFrame(rows, columns=["trial", "reference", "method", "MAE"]).set_index(["trial", "reference", "method"]).MAE


unperturbed_path, trials_path, mae_out, *conditions = sys.argv[1:]
trials = ["cohface_" + t.replace("/", "_") for t in open(trials_path).read().split()]
unperturbed = trial_mae(unperturbed_path, trials)
perturbed = {label: trial_mae(path, trials) for label, path in (c.split("=", 1) for c in conditions)}
pd.concat({"none": unperturbed, **perturbed}, names=["condition"]).reset_index().to_csv(mae_out, index=False, float_format="%.4f")
print(f"{len(trials)} trials, one per subject; change = perturbed - unperturbed trial MAE (bpm); "
      f"worse = subjects whose MAE rises by more than {WORSE_BPM} bpm")
for r in REFERENCES:
    print(f"\nreference {r}")
    print(f"{'condition':18s}" + "".join(f"{m:>30s}" for m in METHODS.values()))
    base = unperturbed.xs(r, level="reference")
    print(f"{'none':18s}" + "".join(f"{base.xs(m, level='method').median():>30.3f}" for m in METHODS.values()))
    for label, scores in perturbed.items():
        scores = scores.xs(r, level="reference")
        cells = []
        for m in METHODS.values():
            after, before = scores.xs(m, level="method"), base.xs(m, level="method")
            change = (after - before).dropna()
            cells.append(f"{after.median():.3f} ({np.median(change):+.3f}, {int((change > WORSE_BPM).sum())}/{change.size})")
        print(f"{label:18s}" + "".join(f"{c:>30s}" for c in cells))
