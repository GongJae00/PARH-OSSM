"""Copy MAHNOB final PARH pickles with the reference placed at the stimulus onset instead of the BDF end.

The loader places the video at the end of the BDF record (tail alignment). Every MAHNOB-HCI BDF
marks the stimulus onset at 30.0 s on its Status channel, and the waveform evidence on the
development subjects (evaluation/waveform/onset_alignment_waveform.py) puts the video start there for
most subjects. The other trigger-defined placement keeps the same number of belt samples, starting
at the onset; a belt record that ends before the video does yields fewer reference samples, and the
window rates then score only the windows that lie inside it. Only `gt` changes; the estimates,
which never read the reference, are copied unchanged.

usage: onset_reference.py <final data dir> <out dir> <MAHNOB root> [sessions csv with a trial column]
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from pyedflib import EdfReader

# The first Status edge of the BDF, which onset_alignment_waveform.py detects, is at 30.0 s in every one of the
# 488 common sessions (checked on all of them), so the onset is a constant here and the BDF's Status channel is not read.
STIMULUS_ONSET_SEC = 30.0
BELT_CHANNEL = 44

final_dir, out_dir, mahnob_root = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
names = sorted(f for f in os.listdir(final_dir) if f.endswith(".pkl"))
if len(sys.argv) > 4:
    names = [n for n in names if n.removesuffix(".pkl") in set(pd.read_csv(sys.argv[4]).trial)]

for name in names:
    payload = pickle.load(open(os.path.join(final_dir, name), "rb"))
    folder = os.path.join(mahnob_root, name.removesuffix(".pkl").split("_")[1])
    reader = EdfReader(os.path.join(folder, next(f for f in os.listdir(folder) if f.lower().endswith(".bdf"))))
    belt = reader.readSignal(BELT_CHANNEL)
    reader.close()

    start = int(round(STIMULUS_ONSET_SEC * payload["fs_gt"]))
    n_reference = np.asarray(payload["gt"]).size
    payload["gt"] = np.asarray(belt[start:start + n_reference], dtype=np.asarray(payload["gt"]).dtype)
    with open(os.path.join(out_dir, name), "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
