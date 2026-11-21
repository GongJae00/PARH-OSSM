"""Per-window rate of the nine observations' OSSM-KF rate tracks combined without weights.

The PARH readout combines the observations' OSSM-KF tracks (external rate evidence ossm_kfstd) with
reliability weights; this control gives every observation the same weight, as the quality-weighting
comparisons of Nguyen et al. 2026 do. At each frame the rate is the mean of the nine tracks, and each
30-s window with a 1-s stride from t = 0 gets the mean of that rate, as window_rates_three_references.py
windows a track. A second, robust combination (descriptive, added after the declaration) takes each track's
median over the window and then the median over the nine tracks, which is the PARH readout's weighted median
across families without its weights. Neither estimate reads the reference, so its windows serve either belt
placement.

usage: equal_weight_track_rates.py <frozen data dir with the nine-family pickles> [workers] > <dataset>_equal_weight_track_rates.csv
"""
import csv
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

OBSERVATION_TRACKS = ["of_farneback__kfstd", "of_disp_bridge__kfstd", "dof__kfstd", "dof_disp_bridge__kfstd",
                      "profile1d_linear__kfstd", "profile1d_quadratic__kfstd", "profile1d_cubic__kfstd",
                      "profile1d_consensus__kfstd", "torso_shift__kfstd"]
WIN, STRIDE = 30, 1


def equal_weight_rates(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    estimates = {e["method"]: e["estimate"] for e in payload["estimates"]}
    tracks = [np.asarray(estimates[name]["track_hz"], dtype=float).reshape(-1) for name in OBSERVATION_TRACKS]
    n_frames = min(track.size for track in tracks)
    track_bpm = 60.0 * np.vstack([track[:n_frames] for track in tracks])
    mean_track_bpm = np.nanmean(track_bpm, axis=0)

    fps = float(payload["fps"])
    win, stride = int(round(WIN * fps)), int(round(STRIDE * fps))
    trial = os.path.basename(path).removesuffix(".pkl")
    return [(trial, w, np.nanmean(mean_track_bpm[s:s + win]), np.nanmedian(np.nanmedian(track_bpm[:, s:s + win], axis=1)))
            for w, s in enumerate(range(0, n_frames - win + 1, stride))]


if __name__ == "__main__":
    data_dir = sys.argv[1]
    names = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))
    out = csv.writer(sys.stdout)
    out.writerow(["trial", "window", "equal_weight_track_bpm", "median_track_bpm"])
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as pool:
        for rows in pool.imap(equal_weight_rates, [os.path.join(data_dir, n) for n in names]):
            out.writerows([(t, w, f"{mean:.4f}", f"{median:.4f}") for t, w, mean, median in rows])
