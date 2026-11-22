"""How far are the observations' errors from the independence the filter assumes?

The multichannel filter weighs each observation with a diagonal measurement covariance R, which
assumes that the errors of two observations at the same instant are unrelated. Before fusing, it
aligns every observation to its anchor with the lag (within LAG_MAX_SEC) and scale that best match
their breathing-band signals (multichannel_fusion._relative_equation). Two observations are therefore
compared in the filter's frame: at the lag their breathing-band cross-correlation picks.

Above the breathing band the chest moves too little to give a common respiratory signal, so the
correlation of two observations there estimates the correlation of their errors, the off-diagonal of
R that the filter sets to zero; part of it is real non-respiratory motion that every image
measurement of the same region shares. The breathing-band correlation mixes the shared respiratory
signal with any shared error. Each is given at lag zero and at the filter's lag, where the breathing
band gives the magnitude of the correlation, as the alignment picks the lag by it.

Correlation at one lag cannot separate a duplicate from a second measurement of the same breathing,
since both carry the breathing signal. The magnitude-squared coherence over the breathing band can:
it is the share of one observation's power at each frequency that a linear filter of the other
explains, whatever the phase. An observation computed from another by a linear filter, such as an
integral, has coherence near 1 with it; a second measurement with errors of its own falls below 1
by the share of those errors.

Per trial, each raw observation is band-passed and robust-scaled as the filter does
(multichannel_fusion._bandpass_z); the table gives the median over trials and its interquartile range.

usage: observation_error_correlation.py <data dir> <out csv> [split csv: development rows only] [workers]
"""
import os
import pickle
import sys
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import coherence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from estimate.readout.multichannel_fusion import _bandpass_z, _overlap, _relative_equation

BREATHING_HZ = (0.08, 0.5)
ABOVE_BREATHING_HZ = (0.7, 2.0)
COHERENCE_SEGMENT_SEC = 15.0
LAG_MAX_SEC = 4.0  # the materializer's --lag-max-sec default, the bound of its channel alignment
OBSERVATIONS = ["of_farneback", "of_disp_bridge", "DoF", "dof_disp_bridge", "profile1D linear",
                "profile1D quadratic", "profile1D cubic", "profile1d_consensus", "torso_shift"]
COLUMNS = ["breathing coherence", "breathing, lag 0", "above, lag 0", "breathing, filter lag", "above, filter lag",
           "filter lag s"]


def trial_correlations(path):
    with open(path, "rb") as f:
        record = pickle.load(f)
    fs = float(record["fps"])
    raw = {e["method"]: np.asarray(e["estimate"]["signal_hat"], dtype=float).ravel()
           for e in record["estimates"] if e["method"] in OBSERVATIONS}
    n = min(x.size for x in raw.values())
    breathing = {name: _bandpass_z(x[:n], fs, *BREATHING_HZ) for name, x in raw.items()}
    above = {name: _bandpass_z(x[:n], fs, *ABOVE_BREATHING_HZ) for name, x in raw.items()}

    rows = []
    for a, b in combinations(OBSERVATIONS, 2):
        alignment = _relative_equation(breathing[a], breathing[b], fs, LAG_MAX_SEC)
        lag = int(alignment["lag_samples"])
        above_b, above_a = _overlap(above[b], above[a], lag)
        freq, msc = coherence(raw[a][:n], raw[b][:n], fs=fs, nperseg=int(COHERENCE_SEGMENT_SEC * fs))
        in_band = (freq >= BREATHING_HZ[0]) & (freq <= BREATHING_HZ[1])
        rows.append((a, b, msc[in_band].mean(),
                     np.corrcoef(breathing[a], breathing[b])[0, 1], np.corrcoef(above[a], above[b])[0, 1],
                     abs(alignment["corr"]), np.corrcoef(above_a, above_b)[0, 1], lag / fs))
    return rows


if __name__ == "__main__":
    data_dir, out_csv = sys.argv[1:3]
    names = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))
    if len(sys.argv) > 3 and sys.argv[3] != "-":
        split = pd.read_csv(sys.argv[3])
        development = set(split.loc[split.split == "development", "trial"])
        names = [n for n in names if n.removesuffix(".pkl") in development]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    with Pool(workers) as pool:
        rows = pool.map(trial_correlations, [os.path.join(data_dir, n) for n in names])
    table = pd.DataFrame([r for trial_rows in rows for r in trial_rows], columns=["a", "b", *COLUMNS])
    table.to_csv(out_csv, index=False)

    print(f"{len(names)} trials; median correlation over trials [interquartile range]")
    print(f"{'':42s}" + "".join(f"{c:>22s}" for c in COLUMNS))
    for (a, b), g in table.groupby(["a", "b"], sort=False):
        cells = [f"{g[c].median():6.2f} [{g[c].quantile(0.25):5.2f},{g[c].quantile(0.75):5.2f}]" for c in COLUMNS[:5]]
        cells.append(f"{g[COLUMNS[5]].median():6.2f}")
        print(f"  {a:20s}{b:20s}" + "".join(f"{c:>22s}" for c in cells))
