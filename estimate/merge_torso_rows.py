"""Add the torso_shift rows to copies of the session pickles (the nine-family bank).

The base pickles hold the eight observations and their per-observation OSSM-KF and PARH rows; a
run of observations/extract.py with only the torso_shift methods (observations/run_shards.py with a torso cache folder)
holds the same three rows for the new observation. Appending them to a copy leaves every other row
unchanged, so materializing PARH on the copies differs from the reported bank by the torso family
alone (the reverse of leave_one_out_data.py).

usage: merge_torso_rows.py <base data dir> <torso run data dir> <out data dir> [sessions file]
"""
import os
import pickle
import sys

TORSO_ROWS = {"torso_shift", "torso_shift__kfstd", "torso_shift__parh_ossm"}

base_dir, torso_dir, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
names = sorted(f for f in os.listdir(torso_dir) if f.endswith(".pkl"))
if len(sys.argv) > 4:
    sessions = {f"mahnob_{s.strip()}.pkl" for s in open(sys.argv[4]) if s.strip()}
    names = [n for n in names if n in sessions]
for name in names:
    payload = pickle.load(open(os.path.join(base_dir, name), "rb"))
    torso = pickle.load(open(os.path.join(torso_dir, name), "rb"))
    rows = [row for row in torso["estimates"] if row["method"] in TORSO_ROWS]
    payload["estimates"] = [row for row in payload["estimates"] if row["method"] not in TORSO_ROWS] + rows
    with open(os.path.join(out_dir, name), "wb") as f:
        pickle.dump(payload, f)
print(len(names), "sessions written to", out_dir)
