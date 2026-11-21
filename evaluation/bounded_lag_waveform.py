"""Aligned waveform CCC with the lag search bounded, against a null of other subjects' references.

The evaluation aligns an estimate to the belt by the argmax of the full cross-correlation, so any lag
up to the trial length can be chosen. Here the same alignment is taken over |lag| <= MAX_LAG_SEC only: half of a 4-s breath, so
the alignment can correct a delay but cannot jump to another breath. Each estimate is also aligned,
by the same rule, to the references of NULL_REFERENCES trials of other subjects; their median CCC is
the level a waveform reaches against an unrelated breathing trace.

Waveforms as the evaluation scores them: PARH the smoothed z_full, OSSM-KF and direct P1D
the signal_hat of the P1D-quadratic heads. Every method uses the reference in the final PARH pickle
(tail-aligned on MAHNOB), band-passed and z-scored as the evaluation does.

usage: bounded_lag_waveform.py <base data dir> <final data dir> <MAHNOB root or "-"> [workers] > <dataset>_bounded_lag_waveform.csv
<base data dir> holds the pickles of observations/extract.py, <final data dir> those of estimate/materialize.sh.
"""
import csv
import os
import pickle
import re
import sys
import warnings
from multiprocessing import Pool

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluation.belt_reference import filter_RW, recorded_belt

MIN_HZ, MAX_HZ = 0.08, 0.5
MAX_LAG_SEC = 2.0
NULL_REFERENCES = 10
WAVEFORMS = {"parh": None, "ossm_kf": "profile1d_quadratic__kfstd", "direct_p1d": "profile1D quadratic"}


def concordance_correlation_coefficient(series_a, series_b):
    """Compute Lin's concordance correlation coefficient with robust guards."""
    arr_a = np.asarray(series_a, dtype=np.float64)
    arr_b = np.asarray(series_b, dtype=np.float64)
    finite_mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    if np.count_nonzero(finite_mask) < 2:
        return np.nan
    a = arr_a[finite_mask]
    b = arr_b[finite_mask]
    eps = 1e-6
    std_a = np.std(a, dtype=np.float64)
    std_b = np.std(b, dtype=np.float64)
    if (not np.isfinite(std_a)) or (std_a < eps) or (not np.isfinite(std_b)) or (std_b < eps):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        corr_matrix = np.corrcoef(a, b)
    if corr_matrix.shape[0] < 2 or not np.isfinite(corr_matrix[0, 1]):
        return np.nan
    rho = corr_matrix[0, 1]
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a)
    var_b = np.var(b)
    numerator = 2 * rho * std_a * std_b
    denominator = var_a + var_b + (mean_a - mean_b) ** 2
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return numerator / denominator


def waveform_ccc(sig_est, sig_gt):
    """CCC between estimated and ground-truth waveforms (sample-level).

    This is for z_full evaluation, NOT for rate BPM sequences.
    Signals should be at the same sampling rate and aligned.

    Returns float CCC in [-1, 1] or NaN.
    """
    a = np.asarray(sig_est, dtype=np.float64).flatten()
    b = np.asarray(sig_gt, dtype=np.float64).flatten()
    n = min(len(a), len(b))
    if n < 2:
        return float('nan')
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 2:
        return float('nan')
    return float(concordance_correlation_coefficient(a[mask], b[mask]))


def zscored_band(x, fs):
    filtered = filter_RW(np.asarray(x, dtype=float).reshape(-1), fs, lo=MIN_HZ, hi=MAX_HZ).reshape(-1)
    return (filtered - filtered.mean()) / (filtered.std() + 1e-9)


def aligned_ccc(estimate, fps, reference, fs_gt, max_lag_sec):
    """CCC after shifting the estimate (resampled to fs_gt) by the correlation-maximising lag; None = unbounded."""
    t = np.arange(round(estimate.size * fs_gt / fps)) / fs_gt
    est = np.interp(t, np.arange(estimate.size) / fps, estimate)
    correlation = signal.correlate((est - est.mean()) / est.std(), (reference - reference.mean()) / reference.std())
    lags = signal.correlation_lags(est.size, reference.size)
    if max_lag_sec is not None:
        inside = np.abs(lags) <= max_lag_sec * fs_gt
        correlation, lags = correlation[inside], lags[inside]
    lag = lags[np.argmax(correlation)]
    est_part, ref_part = (est[lag:], reference) if lag >= 0 else (est, reference[-lag:])
    n = min(est_part.size, ref_part.size)
    return waveform_ccc(est_part[:n], ref_part[:n]), lag / fs_gt


def load(paths):
    base_path, final_path = paths
    with open(final_path, "rb") as f:
        final = pickle.load(f)
    with open(base_path, "rb") as f:
        base = {e["method"]: e["estimate"] for e in pickle.load(f)["estimates"]}
    fps, fs_gt = float(final["fps"]), float(final["fs_gt"])
    parh = final["estimates"][0]["estimate"]
    smoothed = parh.get("z_full_smoothed")
    waveforms = {"parh": smoothed if smoothed is not None else parh["z_full"]}
    for method, key in WAVEFORMS.items():
        if key is not None:
            waveforms[method] = base[key]["signal_hat"]
    waveforms = {m: zscored_band(w, fps) for m, w in waveforms.items() if np.asarray(w).size >= fps * 30}
    return waveforms, fps, zscored_band(recorded_belt(final["gt"]), fs_gt), fs_gt


def trial_rows(job):
    trial, null_trials = job
    waveforms, fps, reference, fs_gt = LOADED[trial]
    null_references = [LOADED[t][2] for t in null_trials]
    rows = []
    for method, waveform in waveforms.items():
        ccc_unbounded, lag_unbounded = aligned_ccc(waveform, fps, reference, fs_gt, None)
        ccc_bounded, lag_bounded = aligned_ccc(waveform, fps, reference, fs_gt, MAX_LAG_SEC)
        null_unbounded = np.median([aligned_ccc(waveform, fps, r, fs_gt, None)[0] for r in null_references])
        null_bounded = np.median([aligned_ccc(waveform, fps, r, fs_gt, MAX_LAG_SEC)[0] for r in null_references])
        rows.append((trial, method, ccc_unbounded, lag_unbounded, null_unbounded, ccc_bounded, lag_bounded, null_bounded))
    return rows


if __name__ == "__main__":
    base_root, final_root, mahnob_root = sys.argv[1], sys.argv[2], sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    trials = sorted(f.removesuffix(".pkl") for f in os.listdir(final_root) if f.endswith(".pkl"))
    if mahnob_root != "-":
        subject_of = {f"mahnob_{s}": int(re.match(r"Part_(\d+)_", b)[1])
                      for s in os.listdir(mahnob_root) for b in os.listdir(os.path.join(mahnob_root, s))
                      if b.lower().endswith(".bdf")}
    else:
        subject_of = {t: int(t.split("_")[1]) for t in trials}

    # Each pickle is read once; the scoring workers inherit LOADED through fork.
    with Pool(workers) as pool:
        LOADED = dict(zip(trials, pool.map(load, [(os.path.join(base_root, t + ".pkl"),
                                                   os.path.join(final_root, t + ".pkl")) for t in trials])))

    # Null references: the NULL_REFERENCES next other-subject trials in a fixed shuffled order.
    shuffled = [trials[i] for i in np.random.default_rng(0).permutation(len(trials))]
    jobs = []
    for k, trial in enumerate(shuffled):
        others = [t for t in shuffled[k + 1:] + shuffled[:k] if subject_of[t] != subject_of[trial]]
        jobs.append((trial, others[:NULL_REFERENCES]))

    out = csv.writer(sys.stdout)
    out.writerow(["trial", "subject", "method", "ccc_unbounded", "lag_unbounded_sec", "null_ccc_unbounded",
                  "ccc_bounded", "lag_bounded_sec", "null_ccc_bounded"])
    with Pool(workers) as pool:
        for rows in pool.imap_unordered(trial_rows, jobs):
            out.writerows([(t, subject_of[t], m, *(f"{v:.4f}" for v in values)) for t, m, *values in rows])
            sys.stdout.flush()
