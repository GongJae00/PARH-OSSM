"""Compute the torso displacement observation (torso_shift in observations/chest_motion.py) for every trial, outside
the dataset tree.

The pipeline reads observation caches from the trial folder of the video, which for the original
datasets is read-only; observations/run_shards.py copies obs_* files into its mirror folders, so the caches
are written here as <out dir>/<dataset>_<trial>.npy and copied into the mirrors as obs_torso_shift.npy (--torso-cache).
COHFACE is measured on the original data.avi, as in the reported runs, although the other observations read
data.cfr.avi, its constant-frame-rate copy, which is one frame longer.

usage: torso_shift_cache.py <MAHNOB|COHFACE> <dataset root> <out dir> [workers] [file of trial ids]
"""
import glob
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from observations.chest_motion import torso_shift
from observations.run_shards import list_trials


def video_path(dataset, folder):
    if dataset == "MAHNOB":
        return sorted(glob.glob(os.path.join(folder, "*.avi")))[0]
    return os.path.join(folder, "data.avi")


def write_cache(job):
    dataset, folder, out_path = job
    np.save(out_path, torso_shift(video_path(dataset, folder)).astype(np.float32))
    return out_path


if __name__ == "__main__":
    dataset, root, out_dir = sys.argv[1:4]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    trials = list_trials(dataset, root)
    if len(sys.argv) > 5:
        selected = set(open(sys.argv[5]).read().split())
        trials = [t for t in trials if t in selected]
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(dataset, os.path.join(root, t), os.path.join(out_dir, f"{dataset.lower()}_{t.replace('/', '_')}.npy"))
            for t in trials]
    jobs = [job for job in jobs if not os.path.exists(job[2])]
    cv2.setNumThreads(1)
    with Pool(workers) as pool:
        for _ in pool.imap(write_cache, jobs):
            pass
