"""The observation bank of PARH-OSSM, defined once.

Extraction computes nine observations. The bank keeps one observation per image measurement; the others are dropped
from the pickles before the reliability priors are fitted and the multichannel filter runs. To change the bank, edit
BANK and ROUTES below and the expected values in tests/test_bank.py; every script that needs the labels, the drop
list, or the route exclusions reads them from here.

usage: bank.py <labels | drop | routes>   prints the bank labels, the dropped labels (comma-separated), or the route
exclusion names (space-separated)
"""
import sys

# label of each extracted observation and the name of its OSSM-KF and PARH rows (the materializer's DEFAULT_METHODS)
NINE = {
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
# the bank: observation label and the name the figures and tables give it
BANK = {
    "of_farneback": "OF",
    "DoF": "DoF",
    "dof_disp_bridge": "DoF bridge",
    "profile1D quadratic": "P1D",
    "torso_shift": "Torso displacement",
}
# the route exclusions of the bank: name, labels left out, row name in the Supplementary
ROUTES = {
    "no_of": (["of_farneback"], "OF left out"),
    "no_dof": (["DoF", "dof_disp_bridge"], "DoF and DoF bridge left out"),
    "no_p1d": (["profile1D quadratic"], "P1D quadratic left out"),
    "no_torso": (["torso_shift"], "Torso displacement left out"),
}
DROPPED = [label for label in NINE if label not in BANK]

if __name__ == "__main__":
    what = sys.argv[1]
    print({"labels": ",".join(BANK), "drop": ",".join(DROPPED), "routes": " ".join(ROUTES)}[what])
