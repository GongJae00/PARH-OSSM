"""Compute the torso displacement observation (torso_shift in observations/image/chest_motion.py) of every MAHNOB-HCI
session, outside the dataset tree.

The pipeline reads observation caches from the session folder of the video, which for the original dataset is
read-only; observations/run_shards.py copies obs_* files into its mirror folders, so the caches are written here as
<out dir>/mahnob_<session>.npy and copied into the mirrors as obs_torso_shift.npy (--torso-cache). COHFACE needs no
cache: its torso displacement is computed inside the extraction, from data.cfr.avi like the other observations.

usage: torso_shift_cache.py <MAHNOB-HCI root> <out dir> [workers] [file of session ids]
"""
import glob
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from observations.image.chest_motion import torso_shift
from observations.run_shards import list_trials


def write_cache(job):
    folder, out_path = job
    np.save(out_path, torso_shift(sorted(glob.glob(os.path.join(folder, "*.avi")))[0]).astype(np.float32))
    return out_path


if __name__ == "__main__":
    root, out_dir = sys.argv[1:3]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    trials = list_trials("MAHNOB", root)
    if len(sys.argv) > 4:
        selected = set(open(sys.argv[4]).read().split())
        trials = [t for t in trials if t in selected]
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(os.path.join(root, t), os.path.join(out_dir, f"mahnob_{t}.npy")) for t in trials]
    jobs = [job for job in jobs if not os.path.exists(job[1])]
    cv2.setNumThreads(1)
    with Pool(workers) as pool:
        for _ in pool.imap(write_cache, jobs):
            pass
