"""Per-window rates of the median spectral fusion of the bank's observations and of torso displacement alone.

The bank is whichever of the nine observations the pickles hold, torso displacement included: the three observations
of PARH-OSSM, or the nine of the extraction. The column median_fusion_bank_bpm is the median over that bank,
median_fusion_chest_bpm over its chest observations. The unweighted multichannel baseline is this per-window median of
the observations' Welch-peak rates, over the same inputs as the method. Torso displacement alone, scored as direct P1D
is scored, shows what that observation gives without any fusion.

Each observation is the signal_hat of its plain estimate in the per-observation pickle; each gets the
Welch-peak rate per 30-s window with a 1-s stride from t = 0, as window_rates_three_references.py
scores direct P1D. No reference is read, so the windows serve either belt placement.

usage: observation_fusion_window_rates.py <per-observation data dir> [workers] > <dataset>_observation_fusion_window_rates.csv
"""
import csv
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from window_rates_three_references import welch_rates

CHEST_OBSERVATIONS = ["of_farneback", "of_disp_bridge", "DoF", "dof_disp_bridge", "profile1D linear",
                      "profile1D quadratic", "profile1D cubic", "profile1d_consensus"]
TORSO = "torso_shift"


def trial_rates(path):
    with open(path, "rb") as f:
        stored = pickle.load(f)
    fps = float(stored["fps"])
    signals = {e["method"]: np.asarray(e["estimate"]["signal_hat"], dtype=float).reshape(-1)
               for e in stored["estimates"] if e["method"] in CHEST_OBSERVATIONS + [TORSO]}
    bank = [name for name in CHEST_OBSERVATIONS + [TORSO] if name in signals]
    rates = {name: welch_rates(signals[name], fps) if signals[name].size and signals[name].std() > 0 else np.array([])
             for name in bank}
    n = max(r.size for r in rates.values())
    table = np.full((len(bank), n), np.nan)
    for row, name in enumerate(bank):
        table[row, :rates[name].size] = rates[name]

    chest_rows = [row for row, name in enumerate(bank) if name in CHEST_OBSERVATIONS]
    chest_median = np.nanmedian(table[chest_rows], axis=0) if chest_rows else np.full(n, np.nan)
    bank_median = np.nanmedian(table, axis=0)
    torso = table[bank.index(TORSO)] if TORSO in bank else np.full(n, np.nan)
    trial = os.path.basename(path).removesuffix(".pkl")
    return [(trial, w, bank_median[w], chest_median[w], torso[w]) for w in range(n)]


if __name__ == "__main__":
    root, workers = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
    paths = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.endswith(".pkl")]
    out = csv.writer(sys.stdout)
    out.writerow(["trial", "window", "median_fusion_bank_bpm", "median_fusion_chest_bpm", "direct_torso_bpm"])
    with Pool(workers) as pool:
        for rows in pool.imap(trial_rates, paths):
            out.writerows((t, w, *(f"{v:.4f}" for v in values)) for t, w, *values in rows)
