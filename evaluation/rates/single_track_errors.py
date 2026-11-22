"""Rate error of each observation tracked alone (Supplementary Table 9).

Each observation's OSSM-KF rate track (the `__kfstd` estimate in the per-trial pickles of the bank, the input of
the PARH readout) is windowed as equal_weight_track_rates.py windows a track, 30-s windows with a 1-s stride from
t = 0, and scored as main_tables.py scores a method: the trial MAE over the windows of each reference, for the
trials with at least 10 windows and a direct P1D rate. PARH-OSSM and OSSM-KF are read from the same window-rate
file for comparison. The script re-runs no method; on MAHNOB-HCI it was added after the test participants were
scored.

usage: single_track_errors.py <bank data dir> <window rates csv> [split csv] [development|test]
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

TRACKS = ["of_farneback__kfstd", "profile1d_quadratic__kfstd", "torso_shift__kfstd"]
REFERENCES = ["hilbert", "welch", "count"]
WIN_S, STRIDE_S, MIN_WINDOWS = 30, 1, 10


def window_rates(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    estimates = {e["method"]: e["estimate"] for e in payload["estimates"]}
    fps = float(payload["fps"])
    win, stride = int(round(WIN_S * fps)), int(round(STRIDE_S * fps))
    tracks = {name: 60.0 * np.asarray(estimates[name]["track_hz"], dtype=float).reshape(-1)
              for name in TRACKS if name in estimates}
    n_frames = min(track.size for track in tracks.values())
    trial = os.path.basename(path).removesuffix(".pkl")
    return [(trial, window, *[np.nanmean(tracks[name][start:start + win]) if name in tracks else np.nan
                              for name in TRACKS])
            for window, start in enumerate(range(0, n_frames - win + 1, stride))]


def main():
    data_dir, windows_csv = sys.argv[1], sys.argv[2]
    paths = [os.path.join(data_dir, name) for name in sorted(os.listdir(data_dir)) if name.endswith(".pkl")]
    with Pool(8) as pool:
        rows = [row for trial_rows in pool.imap(window_rates, paths) for row in trial_rows]
    tracks = pd.DataFrame(rows, columns=["trial", "window", *TRACKS])

    windows = pd.read_csv(windows_csv)
    if len(sys.argv) > 4:
        split = pd.read_csv(sys.argv[3])
        windows = windows[windows.trial.isin(split.trial[split.split == sys.argv[4]])]
    windows = windows[windows.trial.isin(windows.dropna(subset=["direct_p1d_welch_bpm"]).trial.unique())]
    windows = windows.merge(tracks, on=["trial", "window"], how="left")

    for method in TRACKS + ["parh_track_bpm", "ossm_kf_track_bpm"]:
        cells = []
        for reference in REFERENCES:
            column = f"{reference}_reference_bpm"
            scored = windows.dropna(subset=[method, column])
            scored = scored[scored.groupby("trial").trial.transform("size") >= MIN_WINDOWS]
            trial_mae = (scored[method] - scored[column]).abs().groupby(scored.trial).mean()
            cells.append(f"{reference}: med {trial_mae.median():.4f} p90 {trial_mae.quantile(0.9):.4f} "
                         f"n {trial_mae.size}")
        print(method, " | ".join(cells))


if __name__ == "__main__":
    main()
