"""Bounded-lag waveform CCC against phase-randomized copies of each trial's own belt (the COHFACE null question).

bounded_lag_waveform.py takes its null from other subjects' belts. On COHFACE that null is lower for PARH
than for OSSM-KF (median 0.162 against 0.183), which raises PARH's excess over the null. An other-subject
belt differs from the trial's belt in its rate as well as in its timing, so a waveform that keeps closer to
one rate can score lower against it without following the breath any better. A phase-randomized copy keeps
the belt's amplitude spectrum, and so its rate content, and randomizes only the timing: the DFT phases of
the band-passed, z-scored belt are replaced by uniform random phases (Hermitian symmetry kept, so the copy
is real), and the copy is z-scored again. Each waveform is scored against SURROGATES copies by the same
bounded-lag alignment, and the median is its null.

usage: phase_randomized_null.py <base data dir> <final data dir> [workers] > <dataset>_phase_randomized_null.csv
"""
import csv
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from evaluation.waveform.bounded_lag_waveform import load, aligned_ccc, MAX_LAG_SEC

SURROGATES = 10


def phase_randomized(reference, rng):
    spectrum = np.fft.rfft(reference)
    phases = rng.uniform(0, 2 * np.pi, spectrum.size)
    phases[0] = 0.0
    if reference.size % 2 == 0:
        phases[-1] = 0.0
    surrogate = np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=reference.size)
    return (surrogate - surrogate.mean()) / surrogate.std()


def trial_rows(paths):
    waveforms, fps, reference, fs_gt = load(paths)
    trial = os.path.basename(paths[1]).removesuffix(".pkl")
    rng = np.random.default_rng(zlib.crc32(trial.encode()))
    surrogates = [phase_randomized(reference, rng) for _ in range(SURROGATES)]
    rows = []
    for method, waveform in waveforms.items():
        matched = aligned_ccc(waveform, fps, reference, fs_gt, MAX_LAG_SEC)[0]
        null = np.median([aligned_ccc(waveform, fps, s, fs_gt, MAX_LAG_SEC)[0] for s in surrogates])
        rows.append((trial, method, matched, null))
    return rows


if __name__ == "__main__":
    base_root, final_root = sys.argv[1], sys.argv[2]
    names = sorted(f for f in os.listdir(final_root) if f.endswith(".pkl"))
    out = csv.writer(sys.stdout)
    out.writerow(["trial", "method", "ccc_bounded", "phase_randomized_null_ccc_bounded"])
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        for rows in pool.imap(trial_rows, [(os.path.join(base_root, n), os.path.join(final_root, n)) for n in names]):
            out.writerows([(t, m, f"{a:.4f}", f"{b:.4f}") for t, m, a, b in rows])
