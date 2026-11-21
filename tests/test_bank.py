import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "estimate"))
import bank


def test_bank_holds_the_three_observations_that_measure_the_shift_of_an_image_pattern():
    assert list(bank.BANK) == ["of_farneback", "profile1D quadratic", "torso_shift"]


def test_dropped_observations_are_the_other_six_of_the_nine():
    assert bank.DROPPED == ["of_disp_bridge", "DoF", "dof_disp_bridge", "profile1D linear", "profile1D cubic",
                            "profile1d_consensus"]


def test_route_exclusions_remove_one_measurement_route_of_the_bank_each():
    assert {name: labels for name, (labels, _) in bank.ROUTES.items()} == {
        "no_of": ["of_farneback"], "no_p1d": ["profile1D quadratic"], "no_torso": ["torso_shift"]}
    assert all(set(labels) <= set(bank.BANK) for labels, _ in bank.ROUTES.values())


def test_added_back_run_restores_the_frame_difference_statistics_to_the_bank():
    assert {name: labels for name, (labels, _) in bank.ADDED_BACK.items()} == {"with_dof": ["DoF", "dof_disp_bridge"]}
    assert not set(bank.ADDED_BACK["with_dof"][0]) & set(bank.BANK)
