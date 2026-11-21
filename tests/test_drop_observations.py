import os
import pickle
import subprocess
import sys

ESTIMATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "estimate")

NINE = ["of_farneback", "of_disp_bridge", "DoF", "dof_disp_bridge", "profile1D linear", "profile1D quadratic",
        "profile1D cubic", "profile1d_consensus", "torso_shift"]
BANK = ["of_farneback", "DoF", "dof_disp_bridge", "profile1D quadratic", "torso_shift"]


def estimates(observations):
    families = {"of_farneback": "of_farneback", "of_disp_bridge": "of_disp_bridge", "DoF": "dof",
                "dof_disp_bridge": "dof_disp_bridge", "profile1D linear": "profile1d_linear",
                "profile1D quadratic": "profile1d_quadratic", "profile1D cubic": "profile1d_cubic",
                "profile1d_consensus": "profile1d_consensus", "torso_shift": "torso_shift"}
    return [{"method": method, "estimate": {}} for label in observations
            for method in (label, f"{families[label]}__kfstd", f"{families[label]}__parh_ossm")]


def drop(tmp_path, source_methods, variant):
    source, copy = tmp_path / "source", tmp_path / "copy"
    source.mkdir(exist_ok=True)
    with open(source / "mahnob_1.pkl", "wb") as f:
        pickle.dump({"estimates": source_methods}, f)
    subprocess.run([sys.executable, os.path.join(ESTIMATE, "drop_observations.py"), str(source), str(copy), variant], check=True)
    with open(copy / "mahnob_1.pkl", "rb") as f:
        return [e["method"] for e in pickle.load(f)["estimates"]]


def test_bank_keeps_the_five_observations_with_their_filter_rows(tmp_path):
    kept = drop(tmp_path, estimates(NINE), "bank")

    assert kept == [m["method"] for m in estimates(BANK)]


def test_route_exclusion_removes_a_measurement_route_from_the_bank(tmp_path):
    bank = estimates(BANK)

    assert drop(tmp_path, bank, "no_dof") == [m["method"] for m in estimates(["of_farneback", "profile1D quadratic", "torso_shift"])]
    assert drop(tmp_path, bank, "no_torso") == [m["method"] for m in estimates(BANK[:-1])]


def test_chest_only_keeps_the_eight_chest_observations(tmp_path):
    assert drop(tmp_path, estimates(NINE), "chest_only") == [m["method"] for m in estimates(NINE[:-1])]
