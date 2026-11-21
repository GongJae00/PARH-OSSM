"""Copy session pickles with some observations removed.

The bank of PARH-OSSM is defined in bank.py. Extraction computes nine observations; `bank` removes those the bank
does not hold from every pickle, with each removed observation's
signal row and its OSSM-KF and PARH rows, the rows the materializer reads as that observation's rate track. An added-back
run (the names of bank.ADDED_BACK) removes only the observations that stay out of the enlarged bank. A route
exclusion (the names of bank.ROUTES) is applied to the copy of the bank and removes one measurement
route from it, and `chest_only` removes torso displacement from the nine, which leaves the eight chest observations
of the submitted method. `drop:<label>,<label>,...` removes the observations of the labels given.

usage: drop_observations.py <src data dir> <dst data dir> <bank | chest_only | route exclusion | drop:label,label,...>
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank import ADDED_BACK, DROPPED, NINE, ROUTES

VARIANTS = {"bank": DROPPED, "chest_only": ["torso_shift"], **{name: labels for name, (labels, _) in ROUTES.items()},
            **{name: [x for x in DROPPED if x not in labels] for name, (labels, _) in ADDED_BACK.items()}}

src, dst, variant = sys.argv[1:4]
labels = variant.removeprefix("drop:").split(",") if variant.startswith("drop:") else VARIANTS[variant]
removed = set(labels) | {f"{NINE[label]}__{suffix}" for label in labels for suffix in ("kfstd", "parh_ossm")}
os.makedirs(dst, exist_ok=True)
for name in sorted(f for f in os.listdir(src) if f.endswith(".pkl")):
    with open(os.path.join(src, name), "rb") as f:
        trial = pickle.load(f)
    trial["estimates"] = [e for e in trial["estimates"] if e["method"] not in removed]
    with open(os.path.join(dst, name), "wb") as f:
        pickle.dump(trial, f)
print(f"{variant}: removed {sorted(removed)}")
