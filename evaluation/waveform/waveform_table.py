"""Rows of the waveform table and the PARH-OSSM comparison behind it.

Per method, the bounded-lag CCC (|lag| <= 2 s) of the estimated waveform with the trial's own belt, with the belts
of 10 other-subject trials, and with 10 phase-randomized copies of the trial's own belt, which keep the belt's rate
content and change only its timing. On COHFACE the first two come from bounded_lag_waveform.py and the third from
phase_randomized_null.py, whose own-belt CCCs are identical; on MAHNOB-HCI all three come from
onset_alignment_waveform.py under onset placement, on one span per trial. The table gives the medians over trials;
the excess over the phase-randomized null is compared between PARH-OSSM and each comparator per participant (median
over its trials), by a two-sided Wilcoxon signed-rank test. MAHNOB-HCI is restricted to the participants of the
evaluated split, the test participants by default.

usage: waveform_table.py <records dir> cohface <rows .tex>
       waveform_table.py <records dir> mahnob <rows .tex> [evaluated split, default test]
<records dir> holds cohface_bounded_lag_waveform.csv, cohface_phase_randomized_null.csv,
mahnob_<evaluated split>_alignment_waveform.csv (the outputs of the three scripts above) and mahnob_subject_split.csv.
"""
import os
import sys

import pandas as pd
from scipy.stats import wilcoxon

METHODS = {"direct_p1d": "Direct P1D quadratic", "ossm_kf": "OSSM-KF", "parh": "PARH-OSSM"}
records_dir, dataset, rows_path = sys.argv[1:4]
evaluated = sys.argv[4] if len(sys.argv) > 4 else "test"

if dataset == "cohface":
    scores = pd.read_csv(os.path.join(records_dir, "cohface_bounded_lag_waveform.csv"))
    phase = pd.read_csv(os.path.join(records_dir, "cohface_phase_randomized_null.csv"))
    scores = scores.rename(columns={"null_ccc_bounded": "other_subject_null"}).merge(
        phase.rename(columns={"ccc_bounded": "ccc_phase_run", "phase_randomized_null_ccc_bounded": "phase_randomized_null"}),
        on=["trial", "method"])
    print(f"largest |own-belt CCC difference| between the two runs {(scores.ccc_bounded - scores.ccc_phase_run).abs().max():.4f}")
else:
    scores = pd.read_csv(os.path.join(records_dir, f"mahnob_{evaluated}_alignment_waveform.csv"))
    scores = scores.rename(columns={"ccc_onset": "ccc_bounded", "null_ccc_onset": "other_subject_null",
                                    "phase_randomized_null_ccc_onset": "phase_randomized_null"})
    split = pd.read_csv(os.path.join(records_dir, "mahnob_subject_split.csv"))
    scores = scores[scores.trial.isin(split[split.split == evaluated].trial)]

scores["excess"] = scores.ccc_bounded - scores.phase_randomized_null
print(f"{dataset}: {scores.trial.nunique()} trials, {scores.subject.nunique()} participants")

medians = scores.groupby("method")[["ccc_bounded", "other_subject_null", "phase_randomized_null", "excess"]].median()
trials = scores.groupby("method").trial.nunique()
print(medians.round(3).to_string())
with open(rows_path, "w") as f:
    for method, name in METHODS.items():
        m = medians.loc[method]
        f.write(f"{name} & {m.ccc_bounded:.2f} & {m.other_subject_null:.2f} & {m.phase_randomized_null:.2f} & "
                f"{m.excess:.2f} & {trials[method]} \\\\\n")

per_participant = scores.groupby(["subject", "method"]).excess.median().unstack()
for method in ["direct_p1d", "ossm_kf"]:
    difference = per_participant.parh - per_participant[method]
    print(f"excess over the phase-randomized null, PARH-OSSM minus {METHODS[method]}: participant median "
          f"{difference.median():+.3f}, PARH-OSSM higher in {(difference > 0).sum()}/{len(difference)}, "
          f"p {wilcoxon(difference).pvalue:.2g}")
