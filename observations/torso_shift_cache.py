"""Compute the torso displacement observation (torso_shift in observations/chest_motion.py) for every trial, outside
the dataset tree.

The pipeline reads observation caches from the trial folder of the video, which for the original
datasets is read-only; observations/run_shards.py copies obs_* files into its mirror folders, so the caches
are written here as <out dir>/<trial>.npy and copied into the mirrors as obs_torso_shift.npy (--torso-cache).
MAHNOB-HCI development subjects come first, interleaved across subjects, so that development work
can start before the test subjects finish.

usage: torso_shift_cache.py <MAHNOB|COHFACE> <dataset root> <trials csv with a trial column> <out dir> [workers] [split csv]
"""
import glob
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from observations.chest_motion import torso_shift


def video_path(folder):
    return sorted(glob.glob(os.path.join(folder, "*.avi")))[0] if "MAHNOB" in folder else os.path.join(folder, "data.avi")


def write_cache(job):
    folder, out_path = job
    np.save(out_path, torso_shift(video_path(folder)).astype(np.float32))
    return out_path


if __name__ == "__main__":
    dataset, root, trials_csv, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    trials = pd.DataFrame({"trial": sorted(pd.read_csv(trials_csv).trial.unique())})
    if len(sys.argv) > 6:
        trials = trials.merge(pd.read_csv(sys.argv[6]), on="trial")
        trials["rank"] = trials.groupby("subject").cumcount()
        trials = trials.sort_values(["split", "rank", "subject"])
    if dataset == "MAHNOB":
        folder_of = {t: os.path.join(root, t.removeprefix("mahnob_")) for t in trials.trial}
    else:
        folder_of = {t: os.path.join(root, *t.removeprefix("cohface_").split("_")) for t in trials.trial}
    jobs = [(f, os.path.join(out_dir, f"{t}.npy")) for t, f in folder_of.items() if not os.path.exists(os.path.join(out_dir, f"{t}.npy"))]
    cv2.setNumThreads(1)
    with Pool(int(sys.argv[5]) if len(sys.argv) > 5 else 8) as pool:
        for _ in pool.imap(write_cache, jobs):
            pass
