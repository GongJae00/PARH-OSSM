"""The frozen PARH against its reliability controls.

Scored as confirmatory_test.py scores the frozen method: per trial and reference the MAE over the windows
where both exist, with at least 10 such windows, on the trials where direct P1D is computable; per subject
the median over its trials of (control MAE - frozen MAE), positive when the frozen method is better; the
Wilcoxon signed-rank test over subjects. The twelve comparisons (four controls, three references) form
their own Benjamini-Hochberg family, apart from the six confirmatory tests. The constant predictor
(median trial-mean reference of the development trials) is shown beside them and not tested here.

The computation is compare(), which supplementary_tables.py calls for the family-exclusion table.

usage: reliability_controls_summary.py <frozen window rates csv> <split csv | none> <evaluated split | all>
                                       <name>=<window rates csv>:<column> ...
  with split "none" (COHFACE) every trial is development data and the subject is the COHFACE subject id.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, wilcoxon

REFERENCES = ["hilbert", "welch", "count"]
MIN_WINDOWS = 10


def compare(frozen_path, split_path, evaluated_split, comparator_args):
    """Per-trial MAE of the frozen method, each comparator, and the constant, and the per-reference tests."""
    windows = pd.read_csv(frozen_path)
    windows = windows[windows.trial.isin(windows.dropna(subset=["direct_p1d_welch_bpm"]).trial.unique())]
    windows = windows.rename(columns={"parh_track_bpm": "frozen"})
    controls = []
    for argument in comparator_args:
        name, source = argument.split("=", 1)
        path, column = source.rsplit(":", 1)
        rates = pd.read_csv(path)[["trial", "window", column]].rename(columns={column: name})
        windows = windows.merge(rates, on=["trial", "window"], how="left")
        controls.append(name)

    if split_path == "none":
        windows["subject"] = windows.trial.str.split("_").str[1]
        windows["split"] = "development"
    else:
        split = pd.read_csv(split_path).set_index("trial")
        windows["subject"] = windows.trial.map(split.subject)
        windows["split"] = windows.trial.map(split.split)
    trial_reference = windows[windows.split == "development"].groupby("trial")[[f"{r}_reference_bpm" for r in REFERENCES]].mean()
    for r in REFERENCES:
        windows[f"constant_{r}"] = trial_reference[f"{r}_reference_bpm"].median()
    if evaluated_split != "all":
        windows = windows[windows.split == evaluated_split]

    rows = []
    for (trial, subject), w in windows.groupby(["trial", "subject"]):
        for r in REFERENCES:
            reference = w[f"{r}_reference_bpm"]
            for method, column in [("frozen", "frozen"), *((c, c) for c in controls), ("constant", f"constant_{r}")]:
                valid = reference.notna() & w[column].notna()
                if valid.sum() >= MIN_WINDOWS:
                    rows.append((trial, subject, r, method, np.abs(w.loc[valid, column] - reference[valid]).mean()))
    mae = pd.DataFrame(rows, columns=["trial", "subject", "reference", "method", "MAE"])

    tests = []
    for r in REFERENCES:
        pivot = mae[mae.reference == r].pivot_table(index=["trial", "subject"], columns="method", values="MAE").dropna()
        for comparator in [*controls, "constant"]:
            gain = (pivot[comparator] - pivot.frozen).groupby(level="subject").median()
            tests.append({"reference": r, "comparator": comparator, "trials": len(pivot),
                          "subject median gain": gain.median(), "subjects better": f"{int((gain > 0).sum())}/{gain.size}",
                          "trial median gain": (pivot[comparator] - pivot.frozen).median(), "p": wilcoxon(gain).pvalue})
    tests = pd.DataFrame(tests)
    tested = tests.comparator != "constant"
    tests.loc[tested, f"q (BH, {int(tested.sum())} tests)"] = false_discovery_control(tests.loc[tested, "p"])
    return windows, controls, mae, tests


if __name__ == "__main__":
    windows, controls, mae, tests = compare(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
    print(f"{windows.trial.nunique()} trials, {windows.subject.nunique()} subjects")
    print("\nMedian trial MAE (bpm):")
    print(mae.pivot_table(index="method", columns="reference", values="MAE", aggfunc="median")
          .reindex(["frozen", *controls, "constant"]).round(3).to_string())

    print("\nFrozen PARH against each control (gain = control MAE - frozen MAE; the constant is not tested):")
    print(tests.round(4).to_string(index=False))
