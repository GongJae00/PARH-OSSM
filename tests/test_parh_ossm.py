import numpy as np

from estimate.filters.parh_ossm import oscillator_PARH_OSSM

FS = 20.0
RATE_HZ = 0.3


def three_observations(seconds=120, noise_std=0.5, seed=0):
    t = np.arange(int(seconds * FS)) / FS
    rng = np.random.default_rng(seed)
    breath = np.sin(2 * np.pi * RATE_HZ * t)
    return np.stack([breath + noise_std * rng.standard_normal(t.size) for _ in range(3)])


def test_multichannel_filter_settles_on_the_shared_rate():
    estimate = oscillator_PARH_OSSM().run(three_observations(), FS)

    assert abs(np.median(estimate["track_hz"][int(30 * FS):]) - RATE_HZ) < 0.02


def test_filtered_waveform_has_one_sample_per_frame():
    signals = three_observations(seconds=60)

    estimate = oscillator_PARH_OSSM().run(signals, FS)

    assert estimate["track_hz"].shape == (signals.shape[1],)


NINE_OBSERVATIONS = ["of_farneback", "of_disp_bridge", "DoF", "dof_disp_bridge", "profile1D linear",
                     "profile1D quadratic", "profile1D cubic", "profile1d_consensus", "torso_shift"]


def groups_of(names, key):
    head = oscillator_PARH_OSSM()
    group = head._measurement_group_key if key == "measurement" else head._family_group_key
    groups = {}
    for name in names:
        groups.setdefault(group(name), []).append(name)
    return sorted(len(members) for members in groups.values())


def test_measurement_grouping_counts_each_image_measurement_once():
    # OF and its bridge, DoF and its bridge, the four P1D interpolations, torso displacement
    assert groups_of(NINE_OBSERVATIONS, "measurement") == [1, 2, 2, 4]
    # the construction families of the submitted code join only P1D quadratic and cubic
    assert groups_of(NINE_OBSERVATIONS, "family") == [1, 1, 1, 1, 1, 1, 1, 2]


def test_reported_bank_has_one_observation_per_group_under_both_keys():
    bank = ["of_farneback", "profile1D quadratic", "torso_shift"]
    assert groups_of(bank, "measurement") == groups_of(bank, "family") == [1, 1, 1]
