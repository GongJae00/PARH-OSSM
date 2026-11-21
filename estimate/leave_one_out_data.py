"""Copy session pickles with some observations of the bank removed.

Materializing PARH on the copies (estimate/multichannel_fusion.py with the frozen flags) gives PARH with those observations left out of
the bank and nothing else changed: each removed observation loses its signal row and its per-
observation OSSM-KF and PARH rows, the rows the materializer reads as that observation's rate track.
On MAHNOB-HCI the bank holds the 488 sessions with all eight chest observations
(evaluation/mahnob_common488_sessions.txt). A bank that also holds the 37 sessions without P1D is limited to
those 488 with the sessions file, because the 37 keep too few observations once a family is removed and the
materializer's strict mode stops on them.

usage: leave_one_out_data.py <src_data_dir> <dst_data_dir> <variant> [sessions file]
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
    "no_of": ["of_farneback", "of_disp_bridge"],
    "no_dof": ["DoF", "dof_disp_bridge"],
    "no_p1d": ["profile1D linear", "profile1D quadratic", "profile1D cubic", "profile1d_consensus"],
    "no_bridges": ["of_disp_bridge", "dof_disp_bridge"],
    "no_p1d_cubic": ["profile1D cubic"],
    "no_p1d_consensus": ["profile1d_consensus"],
    "no_torso": ["torso_shift"],
}

src, dst, variant = sys.argv[1], sys.argv[2], sys.argv[3]
removed = set(VARIANTS[variant]) | {f"{FAMILY_OF_LABEL[label]}__{suffix}"
                                    for label in VARIANTS[variant] for suffix in ("kfstd", "parh_ossm")}
os.makedirs(dst, exist_ok=True)
names = sorted(f for f in os.listdir(src) if f.endswith(".pkl"))
if len(sys.argv) > 4:
    sessions = {f"mahnob_{s.strip()}.pkl" for s in open(sys.argv[4]) if s.strip()}
    names = [n for n in names if n in sessions]
for name in names:
    with open(os.path.join(src, name), "rb") as f:
        trial = pickle.load(f)
    trial["estimates"] = [e for e in trial["estimates"] if e["method"] not in removed]
    with open(os.path.join(dst, name), "wb") as f:
        pickle.dump(trial, f)
print(f"{variant}: removed {sorted(removed)}")
