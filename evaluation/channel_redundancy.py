"""How redundant are the nine observations the method reads? (R1-2: several channels look like variants.)

For every pair of observations in a trial this measures two things on the per-observation pickles that
PARH-OSSM reads (the signal_hat of each plain estimate):
- waveform similarity: the largest |normalised cross-correlation| of the band-passed (0.08-0.5 Hz)
  signals within +-3 s, so that P1D, a velocity-like shift (F4), is not penalised for leading the
  displacement-like observations by a quarter cycle;
- rate agreement: the median |difference| of the two observations' 30-s window rates (Welch peak,
  1-s stride, as window_rates_three_references.py scores direct P1D), the quantity the fusion reports.
A pair that is redundant for rate estimation agrees in rate in nearly every window. No reference is read.

usage: channel_redundancy.py <per-observation data dir> [workers] > <dataset>_channel_redundancy.csv
"""
import csv
import os
import pickle
import sys
from itertools import combinations
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from window_rates_three_references import MAX_HZ, MIN_HZ, filter_RW, welch_rates

OBSERVATIONS = ["of_farneback", "of_disp_bridge", "DoF", "dof_disp_bridge", "profile1D linear",
                "profile1D quadratic", "profile1D cubic", "profile1d_consensus", "torso_shift"]
MAX_LAG_SEC = 3.0


def waveform_similarity(a, b, max_lag):
    a = (a - a.mean()) / (a.std() * a.size)
    b = (b - b.mean()) / b.std()
    full = np.correlate(a, b, mode="full")
    centre = a.size - 1
    return np.abs(full[centre - max_lag:centre + max_lag + 1]).max()


def trial_pairs(path):
    with open(path, "rb") as f:
        stored = pickle.load(f)
    fps = float(stored["fps"])
    signals = {e["method"]: np.asarray(e["estimate"]["signal_hat"], dtype=float).reshape(-1) for e in stored["estimates"]}
    observations = [signals[name] for name in OBSERVATIONS]
    n = min(o.size for o in observations)
    observations = [o[:n] for o in observations]

    band = [filter_RW(o, fps, lo=MIN_HZ, hi=MAX_HZ).reshape(-1) if o.std() > 0 else o for o in observations]
    rates = [welch_rates(o, fps) for o in observations]

    rows = []
    for i, j in combinations(range(len(OBSERVATIONS)), 2):
        flat = band[i].std() == 0 or band[j].std() == 0
        similarity = np.nan if flat else waveform_similarity(band[i], band[j], int(round(MAX_LAG_SEC * fps)))
        rate_gap = np.nan if flat else np.median(np.abs(rates[i] - rates[j]))
        rows.append((i, j, similarity, rate_gap))
    return os.path.basename(path).removesuffix(".pkl"), rows


if __name__ == "__main__":
    root, workers = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
    paths = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.endswith(".pkl")]
    out = csv.writer(sys.stdout)
    out.writerow(["trial", "channel_a", "channel_b", "waveform_similarity", "rate_gap_bpm"])
    with Pool(workers) as pool:
        for trial, rows in pool.imap(trial_pairs, paths):
            out.writerows([(trial, OBSERVATIONS[i], OBSERVATIONS[j], f"{s:.4f}", f"{g:.3f}") for i, j, s, g in rows])
