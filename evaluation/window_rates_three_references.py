"""Per-window rates of every method against three reference-rate definitions.

The paper scores spectral estimates (direct P1D) against the Welch peak of the belt window and
track estimates (OSSM-KF, PARH) against the window mean of the belt's Hilbert frequency. Each
definition shares its estimate type with one side of the comparison, so the ranking can follow
the definition. A third reference counts breaths in the belt itself and favours neither type:
Schafer and Kratky's counting method (2008), as Charlton et al. (2016) describe it. In each window
the band-passed belt's local maxima above 0.2 x the 75th percentile of those maxima are the
relevant peaks; a breath is the interval between consecutive relevant peaks that contains exactly
one local minimum below zero; the rate is 60 / the mean duration of the breaths that lie wholly
inside the window. Windows with fewer than two breaths get no count reference.

Windows are 30 s with a 1-s stride from t = 0 on the video time base, as the evaluation pairs them.
The belt trace is the one in the final PARH pickle (tail-aligned on MAHNOB), so every method sees
the same reference samples, up to its last recorded sample (recorded_belt: the fill of the last BDF
data record is not belt).

usage: window_rates_three_references.py <base data dir> <final data dir> [workers] > <dataset>_window_rates_three_references.csv
"""
import csv
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
from scipy.signal import argrelextrema

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluation.belt_reference import filter_RW, recorded_belt, sig_windowing, sig_to_RPM
from evaluation.rate_tracks import _instantaneous_freq_hz, _windowed_track_bpm

WIN, STRIDE, MIN_HZ, MAX_HZ = 30, 1, 0.08, 0.5


def welch_rates(signal, fs):
    windows, _ = sig_windowing(filter_RW(signal, fs, lo=MIN_HZ, hi=MAX_HZ), fs, WIN, stride=STRIDE)
    return sig_to_RPM(windows, fs, int(WIN / 1.5), MIN_HZ, MAX_HZ).reshape(-1)


def track_rates(track_hz, gt_inst_hz, fps, fs_gt):
    return _windowed_track_bpm(np.asarray(track_hz, dtype=float).reshape(-1), gt_inst_hz, fps, fs_gt, WIN, STRIDE)[0]


def breath_count_rate(belt, fs):
    peaks = argrelextrema(belt, np.greater)[0]
    troughs = argrelextrema(belt, np.less)[0]
    if peaks.size < 3:
        return np.nan
    relevant = peaks[belt[peaks] > 0.2 * np.percentile(belt[peaks], 75)]
    low_troughs = troughs[belt[troughs] < 0]
    breaths = [(b - a) / fs for a, b in zip(relevant[:-1], relevant[1:])
               if np.count_nonzero((low_troughs > a) & (low_troughs < b)) == 1]
    return 60.0 / np.mean(breaths) if len(breaths) >= 2 else np.nan


def trial_windows(paths):
    base_path, final_path = paths
    with open(final_path, "rb") as f:
        final = pickle.load(f)
    with open(base_path, "rb") as f:
        base = {e["method"]: e["estimate"] for e in pickle.load(f)["estimates"]}
    gt = recorded_belt(final["gt"])
    fs_gt, fps = float(final["fs_gt"]), float(final["fps"])

    belt = filter_RW(gt, fs_gt, lo=MIN_HZ, hi=MAX_HZ).reshape(-1)
    gt_inst_hz = _instantaneous_freq_hz(belt, fs_gt, MIN_HZ, MAX_HZ)
    n_video = np.asarray(final["estimates"][0]["estimate"]["track_hz"]).size
    hilbert_reference = _windowed_track_bpm(np.ones(n_video), gt_inst_hz, fps, fs_gt, WIN, STRIDE)[1]
    welch_reference = welch_rates(gt, fs_gt)
    win, stride = int(round(WIN * fs_gt)), int(round(STRIDE * fs_gt))
    count_reference = np.array([breath_count_rate(belt[s:s + win], fs_gt)
                                for s in range(0, belt.size - win + 1, stride)])

    p1d = np.asarray(base["profile1D quadratic"]["signal_hat"], dtype=float).reshape(-1)
    estimates = {
        "parh_track": track_rates(final["estimates"][0]["estimate"]["track_hz"], gt_inst_hz, fps, fs_gt),
        "ossm_kf_track": track_rates(base["profile1d_quadratic__kfstd"]["track_hz"], gt_inst_hz, fps, fs_gt),
        "direct_p1d_welch": welch_rates(p1d, fps) if p1d.size else np.array([]),
    }
    n = min(hilbert_reference.size, welch_reference.size, count_reference.size,
            estimates["parh_track"].size, estimates["ossm_kf_track"].size)
    p1d_rates = estimates["direct_p1d_welch"]
    trial = os.path.basename(final_path).removesuffix(".pkl")
    return [(trial, w, hilbert_reference[w], welch_reference[w], count_reference[w],
             estimates["parh_track"][w], estimates["ossm_kf_track"][w],
             p1d_rates[w] if w < p1d_rates.size else np.nan) for w in range(n)]


if __name__ == "__main__":
    base_root, final_root = sys.argv[1], sys.argv[2]
    names = sorted(f for f in os.listdir(final_root) if f.endswith(".pkl"))
    out = csv.writer(sys.stdout)
    out.writerow(["trial", "window", "hilbert_reference_bpm", "welch_reference_bpm", "count_reference_bpm",
                  "parh_track_bpm", "ossm_kf_track_bpm", "direct_p1d_welch_bpm"])
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as pool:
        for rows in pool.imap(trial_windows, [(os.path.join(base_root, n), os.path.join(final_root, n)) for n in names]):
            out.writerows([(t, w, *(f"{v:.4f}" for v in values)) for t, w, *values in rows])
