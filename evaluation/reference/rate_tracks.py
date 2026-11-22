from typing import Tuple

import numpy as np
from scipy import signal as sps


def _instantaneous_freq_hz(sig: np.ndarray, fs: float, min_hz: float, max_hz: float) -> np.ndarray:
    """Approximate instantaneous frequency from analytic phase (Hz)."""
    x = np.asarray(sig, dtype=np.float64).reshape(-1)
    if x.size < 4 or fs <= 0:
        return np.full(x.shape, np.nan, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    analytic = sps.hilbert(x)
    phase = np.unwrap(np.angle(analytic))
    inst = np.diff(phase) / (2.0 * np.pi) * fs
    if inst.size == 0:
        return np.full(x.shape, np.nan, dtype=np.float64)
    inst = np.concatenate([[inst[0]], inst])
    inst = np.clip(inst, min_hz, max_hz)
    return inst.astype(np.float64)


def _windowed_track_bpm(
    est_track_hz: np.ndarray,
    gt_inst_hz: np.ndarray,
    fps_est: float,
    fs_gt: float,
    win_size_sec: float,
    stride_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Window-average estimator/GT frequency tracks on the estimator time base."""
    est_track = np.asarray(est_track_hz, dtype=np.float64).reshape(-1)
    gt_track = np.asarray(gt_inst_hz, dtype=np.float64).reshape(-1)
    if est_track.size < 2 or gt_track.size < 2 or fps_est <= 0 or fs_gt <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    t_est = np.arange(est_track.size, dtype=np.float64) / float(fps_est)
    t_gt = np.arange(gt_track.size, dtype=np.float64) / float(fs_gt)
    gt_interp = np.interp(t_est, t_gt, gt_track)

    win_frames = max(1, int(round(float(win_size_sec) * float(fps_est))))
    stride_frames = max(1, int(round(float(stride_sec) * float(fps_est))))
    if est_track.size < win_frames:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    starts = np.arange(0, est_track.size - win_frames + 1, stride_frames, dtype=np.int64)
    if starts.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    est_bpm = np.asarray(
        [np.nanmean(est_track[s:s + win_frames]) * 60.0 for s in starts],
        dtype=np.float64,
    )
    gt_bpm = np.asarray(
        [np.nanmean(gt_interp[s:s + win_frames]) * 60.0 for s in starts],
        dtype=np.float64,
    )
    # windows stay in place, NaN where a track has no finite sample, so that window w of every method and of every
    # reference is the same 30 s of video
    return est_bpm, gt_bpm
