import numpy as np
import pytest

from estimate.ossm_kf import oscillator_KFstd

FS = 20.0


def noisy_sine(rate_hz, seconds=120, noise_std=0.5, seed=0):
    t = np.arange(int(seconds * FS)) / FS
    return np.sin(2 * np.pi * rate_hz * t) + noise_std * np.random.default_rng(seed).standard_normal(t.size)


@pytest.mark.parametrize("rate_hz", [0.2, 0.3, 0.4])
def test_frequency_track_settles_on_the_rate_of_a_noisy_sine(rate_hz):
    track_hz = oscillator_KFstd().run(noisy_sine(rate_hz), FS)["track_hz"]

    assert abs(np.median(track_hz[int(30 * FS):]) - rate_hz) < 0.02


def test_empty_signal_gives_empty_track():
    assert oscillator_KFstd().run(np.array([]), FS)["track_hz"].size == 0
