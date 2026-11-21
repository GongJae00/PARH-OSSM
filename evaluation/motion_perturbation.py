"""Write COHFACE trial copies whose video carries a controlled non-respiratory perturbation.

The perturbation moves or relights the whole frame, as a shaking camera or flickering light would; it does
not move the subject against a still background, so it stands in for non-respiratory motion in general,
not for a particular kind of body motion. Each kind follows one random process per trial: white noise
band-limited to 0.05-2 Hz, scaled to unit RMS, then to the level. The band overlaps the respiratory band
(0.08-0.5 Hz), where a perturbation competes with the breath instead of being removed by the band-pass.

  translation  horizontal and vertical shifts, independent processes, level = RMS shift per axis in pixels
  rotation     in-plane rotation about the frame centre, level = RMS angle in degrees
  brightness   multiplicative gain 1 + level * process, level = RMS fractional change

Level 0 re-encodes the frames unchanged, so it is the baseline of the same from-video path. Frames are
written losslessly (FFV1) and the reference signal (data.hdf5) is linked, so every observation is computed
afresh from the copy; no observation cache is carried over.

usage: motion_perturbation.py <source root> <out root> <kind> <level> <subject/trial> ...
"""
import os
import sys
import zlib

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

BAND_HZ = (0.05, 2.0)


def unit_process(n_frames, fps, rng):
    sos = butter(2, BAND_HZ, btype="bandpass", fs=fps, output="sos")
    process = sosfiltfilt(sos, rng.standard_normal(n_frames + 2 * int(fps * 10)))[int(fps * 10):][:n_frames]
    return process / np.sqrt(np.mean(process ** 2))


def perturbed_frames(capture, kind, level, rng):
    fps = capture.get(cv2.CAP_PROP_FPS)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first, second = unit_process(n_frames, fps, rng), unit_process(n_frames, fps, rng)

    t = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        k = min(t, n_frames - 1)
        if level == 0:
            yield frame
        elif kind == "translation":
            shift = np.array([[1, 0, level * first[k]], [0, 1, level * second[k]]], dtype=np.float32)
            yield cv2.warpAffine(frame, shift, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        elif kind == "rotation":
            turn = cv2.getRotationMatrix2D((width / 2, height / 2), level * first[k], 1.0)
            yield cv2.warpAffine(frame, turn, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        elif kind == "brightness":
            yield np.clip(frame.astype(np.float32) * (1 + level * first[k]), 0, 255).astype(np.uint8)
        t += 1


def write_trial(source_root, out_root, kind, level, trial):
    src = os.path.join(source_root, trial)
    dst = os.path.join(out_root, "COHFACE", trial)
    os.makedirs(dst, exist_ok=True)
    if not os.path.lexists(os.path.join(dst, "data.hdf5")):
        os.symlink(os.path.join(src, "data.hdf5"), os.path.join(dst, "data.hdf5"))

    capture = cv2.VideoCapture(os.path.join(src, "data.cfr.avi"))
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rng = np.random.default_rng(zlib.crc32(f"{trial}/{kind}".encode()))
    writer = cv2.VideoWriter(os.path.join(dst, "data.cfr.avi"), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
    for frame in perturbed_frames(capture, kind, level, rng):
        writer.write(frame)
    writer.release()
    capture.release()


if __name__ == "__main__":
    source_root, out_root, kind, level = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    for trial in sys.argv[5:]:
        write_trial(source_root, out_root, kind, level, trial)
        print(trial, flush=True)
