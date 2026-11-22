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
