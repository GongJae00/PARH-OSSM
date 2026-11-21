import numpy as np

from evaluation.rate_tracks import _instantaneous_freq_hz, _windowed_track_bpm
from evaluation.window_rates_three_references import breath_count_rate, welch_rates

FS = 32.0
BREATHS_PER_MIN = 15.0


def sine(seconds, rate_hz=BREATHS_PER_MIN / 60.0, fs=FS):
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2 * np.pi * rate_hz * t)


def test_welch_peak_rate_of_a_sine_is_its_frequency():
    rates = welch_rates(sine(90), FS)

    assert rates.size == 60 + 1
    assert np.allclose(rates, BREATHS_PER_MIN, atol=0.5)


def test_breath_count_rate_of_a_sine_is_its_frequency():
    assert abs(breath_count_rate(sine(30), FS) - BREATHS_PER_MIN) < 0.5


def test_breath_count_needs_two_breaths():
    assert np.isnan(breath_count_rate(sine(6), FS))


def test_hilbert_frequency_of_a_sine_is_its_frequency():
    inst_hz = _instantaneous_freq_hz(sine(60), FS, 0.08, 0.5)

    assert np.allclose(inst_hz[int(5 * FS):-int(5 * FS)], BREATHS_PER_MIN / 60.0, atol=0.01)


def test_windowed_track_averages_both_tracks_over_the_same_windows():
    fps = 10.0
    estimate_hz = np.full(600, 0.25)
    reference_hz = np.full(2000, 0.20)

    estimate_bpm, reference_bpm = _windowed_track_bpm(estimate_hz, reference_hz, fps, 33.3, 30.0, 1.0)

    assert estimate_bpm.size == (600 - 300) // 10 + 1
    assert np.allclose(estimate_bpm, 15.0)
    assert np.allclose(reference_bpm, 12.0)


def test_windowed_track_keeps_a_window_without_estimate_in_place():
    fps = 10.0
    estimate_hz = np.full(600, 0.25)
    estimate_hz[:400] = np.nan
    reference_hz = np.full(2000, 0.20)

    estimate_bpm, reference_bpm = _windowed_track_bpm(estimate_hz, reference_hz, fps, 33.3, 30.0, 1.0)

    assert estimate_bpm.size == reference_bpm.size == (600 - 300) // 10 + 1
    assert np.isnan(estimate_bpm[:11]).all() and np.allclose(estimate_bpm[11:], 15.0)
