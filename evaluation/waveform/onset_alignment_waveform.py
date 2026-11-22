"""MAHNOB waveform agreement under two placements of the video on the BDF clock.

Every MAHNOB BDF carries two Status events: the stimulus onset at 30.0 s and the offset about 31 s
before the BDF end; the video carries none. The loader tail-aligns the belt (the video ends with the
BDF), and in the regular sessions the median video length equals stimulus + 31 s, which is also what
a video starting at the stimulus onset would give. The two placements differ by more than 2 s in most
sessions. If one of them is the true timing, the estimated waveforms should agree with the belt above
the other-subject null under that placement and not under the other.

A video longer than stimulus + 31 s runs past the BDF end when placed at the onset, and a shorter
segment raises the maximum over the +-2 s lags. So every comparison of a trial uses the same span:
the first T seconds of the video, T being its overlap with the belt under onset placement. Tail
placement takes the record's reference (BDF channel 44 from bdf_end - video_length, or from 0 when
the BDF is shorter); onset placement takes BDF channel 44 from the onset. Both end at the last recorded
belt sample: the fill of the last BDF data record is not part of the belt (recorded_belt). Each method's waveform (as
in bounded_lag_waveform.py) is scored by the bounded-lag CCC against the trial's own belt and against
the belts of 10 other-subject trials placed by the same rule and cut to their own T. Under onset placement it is
also scored against 10 phase-randomized copies of the trial's own belt over the same span (phase_randomized_null.py),
which keep the belt's rate content and change only its timing.

usage: onset_alignment_waveform.py <base data dir> <final data dir> <MAHNOB root> [workers] > mahnob_onset_alignment_waveform.csv
<base data dir> holds the pickles of observations/extract.py, <final data dir> those of estimate/materialize.sh.
"""
import csv
import os
import zlib
import pickle
import re
import sys
from multiprocessing import Pool

import numpy as np
import pyedflib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from evaluation.reference.belt_reference import recorded_belt
from evaluation.waveform.bounded_lag_waveform import MAX_LAG_SEC, NULL_REFERENCES, WAVEFORMS, aligned_ccc, zscored_band
from evaluation.waveform.phase_randomized_null import SURROGATES, phase_randomized

BELT_CHANNEL = 44
PLACEMENTS = ["tail", "onset"]


def belt_and_onset(folder):
    bdf = next(f for f in os.listdir(folder) if f.lower().endswith(".bdf"))
    reader = pyedflib.EdfReader(os.path.join(folder, bdf))
    belt = recorded_belt(reader.readSignal(BELT_CHANNEL))
    k = reader.getSignalLabels().index("Status")
    status = reader.readSignal(k, digital=True).astype(np.int64) & 0xFFFF
    onset_sec = (np.flatnonzero((status[1:] != 0) & (status[:-1] == 0))[0] + 1) / reader.getSampleFrequency(k)
    reader.close()
    return belt, onset_sec, int(re.match(r"Part_(\d+)_", bdf)[1])


def load(job):
    base_path, final_path, folder = job
    with open(final_path, "rb") as f:
        final = pickle.load(f)
    with open(base_path, "rb") as f:
        base = {e["method"]: e["estimate"] for e in pickle.load(f)["estimates"]}
    fps, fs_gt = float(final["fps"]), float(final["fs_gt"])
    parh = final["estimates"][0]["estimate"]
    smoothed = parh.get("z_full_smoothed")
    raw = {"parh": smoothed if smoothed is not None else parh["z_full"]}
    raw.update({m: base[key]["signal_hat"] for m, key in WAVEFORMS.items() if key is not None})

    belt, onset_sec, subject = belt_and_onset(folder)
    tail_reference = recorded_belt(final["gt"])
    onset = int(round(onset_sec * fs_gt))
    n_video = int(round(np.asarray(raw["parh"]).size / fps * fs_gt))
    span = min(n_video, belt.size - onset, tail_reference.size)
    span_frames = int(span / fs_gt * fps)

    waveforms = {m: zscored_band(np.asarray(w, dtype=float).reshape(-1)[:span_frames], fps)
                 for m, w in raw.items() if np.asarray(w).size >= fps * 30}
    references = {"tail": zscored_band(tail_reference[:span], fs_gt),
                  "onset": zscored_band(belt[onset:onset + span], fs_gt)}
    gap_sec = onset_sec - (belt.size - tail_reference.size) / fs_gt
    return waveforms, fps, references, fs_gt, subject, gap_sec, span / fs_gt


def trial_rows(job):
    trial, null_trials = job
    waveforms, fps, references, fs_gt, subject, gap_sec, span_sec = LOADED[trial]
    rng = np.random.default_rng(zlib.crc32(trial.encode()))
    surrogates = [phase_randomized(references["onset"], rng) for _ in range(SURROGATES)]
    rows = []
    for method, waveform in waveforms.items():
        row = [trial, subject, method, f"{gap_sec:.2f}", f"{span_sec:.2f}"]
        for p in PLACEMENTS:
            matched = aligned_ccc(waveform, fps, references[p], fs_gt, MAX_LAG_SEC)[0]
            null = np.median([aligned_ccc(waveform, fps, LOADED[t][2][p], fs_gt, MAX_LAG_SEC)[0] for t in null_trials])
            row += [f"{matched:.4f}", f"{null:.4f}"]
        phase_null = np.median([aligned_ccc(waveform, fps, s, fs_gt, MAX_LAG_SEC)[0] for s in surrogates])
        rows.append(row + [f"{phase_null:.4f}"])
    return rows


if __name__ == "__main__":
    base_root, final_root, mahnob_root = sys.argv[1], sys.argv[2], sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    trials = sorted(f.removesuffix(".pkl") for f in os.listdir(final_root) if f.endswith(".pkl"))

    # Each pickle and BDF is read once; the scoring workers inherit LOADED through fork.
    with Pool(workers) as pool:
        LOADED = dict(zip(trials, pool.map(load, [(os.path.join(base_root, t + ".pkl"),
                                                   os.path.join(final_root, t + ".pkl"),
                                                   os.path.join(mahnob_root, t.removeprefix("mahnob_")))
                                                  for t in trials])))

    # Null references: the NULL_REFERENCES next other-subject trials in a fixed shuffled order.
    shuffled = [trials[i] for i in np.random.default_rng(0).permutation(len(trials))]
    jobs = []
    for k, trial in enumerate(shuffled):
        others = [t for t in shuffled[k + 1:] + shuffled[:k] if LOADED[t][4] != LOADED[trial][4]]
        jobs.append((trial, others[:NULL_REFERENCES]))

    out = csv.writer(sys.stdout)
    out.writerow(["trial", "subject", "method", "onset_minus_tail_sec", "span_sec",
                  "ccc_tail", "null_ccc_tail", "ccc_onset", "null_ccc_onset", "phase_randomized_null_ccc_onset"])
    with Pool(workers) as pool:
        for rows in pool.imap_unordered(trial_rows, jobs):
            out.writerows(rows)
            sys.stdout.flush()
