"""Copy session pickles with some observations removed.

The bank of PARH-OSSM holds five observations: OF, DoF, the DoF bridge, P1D quadratic, and torso displacement.
Extraction computes nine; `bank` removes the other four from every pickle, with each removed observation's
signal row and its OSSM-KF and PARH rows, the rows the materializer reads as that observation's rate track. A route
exclusion (`no_of`, `no_dof`, `no_p1d`, `no_torso`) is applied to the copy of the bank and removes one measurement
route from it, and `chest_only` removes torso displacement from the nine, which leaves the eight chest observations
of the submitted method. `drop:<label>,<label>,...` removes the observations of the labels given.

usage: drop_observations.py <src data dir> <dst data dir> <bank | chest_only | no_of | no_dof | no_p1d | no_torso | drop:label,label,...>
"""
import os
import pickle
import sys

# Observation family names, as the materializer's DEFAULT_METHODS pairs them with their labels.
FAMILY_OF_LABEL = {
    "of_farneback": "of_farneback",
    "of_disp_bridge": "of_disp_bridge",
    "DoF": "dof",
    "dof_disp_bridge": "dof_disp_bridge",
    "profile1D linear": "profile1d_linear",
    "profile1D quadratic": "profile1d_quadratic",
    "profile1D cubic": "profile1d_cubic",
    "profile1d_consensus": "profile1d_consensus",
    "torso_shift": "torso_shift",
}
VARIANTS = {
    "bank": ["of_disp_bridge", "profile1D linear", "profile1D cubic", "profile1d_consensus"],
    "chest_only": ["torso_shift"],
    "no_of": ["of_farneback"],
    "no_dof": ["DoF", "dof_disp_bridge"],
    "no_p1d": ["profile1D quadratic"],
    "no_torso": ["torso_shift"],
}

src, dst, variant = sys.argv[1:4]
labels = variant.removeprefix("drop:").split(",") if variant.startswith("drop:") else VARIANTS[variant]
removed = set(labels) | {f"{FAMILY_OF_LABEL[label]}__{suffix}" for label in labels for suffix in ("kfstd", "parh_ossm")}
os.makedirs(dst, exist_ok=True)
for name in sorted(f for f in os.listdir(src) if f.endswith(".pkl")):
    with open(os.path.join(src, name), "rb") as f:
        trial = pickle.load(f)
    trial["estimates"] = [e for e in trial["estimates"] if e["method"] not in removed]
    with open(os.path.join(dst, name), "wb") as f:
        pickle.dump(trial, f)
print(f"{variant}: removed {sorted(removed)}")
