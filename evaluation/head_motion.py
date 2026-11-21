"""R1-4a: head motion of each trial, measured from the video without the reference.

Seated participants move mostly by moving the head and upper body together, and both datasets frame the
face large and central, so the face-detection box is tracked through the whole video: MediaPipe face
detection (model_selection=1, the setting torso_shift uses to place its box) every int(fps/4) frames,
about four times a second. The track of each trial is saved, so the measures can be recomputed without
detection.

Primary measure: the 5-95th percentile range of the box centre along each image axis, combined as a
Euclidean norm, in percent of the median face height; percentiles over detections keep a stray detection
from setting the range. Secondary measure: the mean speed of the centre after a 1-s moving average, in
percent of the face height per second, with missed detections linearly interpolated. The share of
samples without a detection is reported beside both, since a face lost during large motion shortens the
range. Trials are those of the first R1-4a analysis (rate_agreement/<prefix>_motion_trials.csv).

usage: head_motion.py <track dir> <jobs> > head_motion_trials.csv
"""
import csv
import os
import sys
from multiprocessing import Pool

import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COHFACE = "/mnt/hdd20t/Dataset/rPPG_Dataset/cohface"
MAHNOB = "/mnt/hdd20t/Dataset/rPPG_Dataset/MAHNOB_HCI"
SAMPLES_PER_SEC = 4


def video_path(trial):
    dataset, *parts = trial.split("_")
    if dataset == "cohface":
        return os.path.join(COHFACE, *parts, "data.cfr.avi")
    session = os.path.join(MAHNOB, parts[0])
    return os.path.join(session, next(f for f in os.listdir(session) if f.endswith(".avi")))  # the loader's choice


def face_track(video):
    capture = cv.VideoCapture(video)
    fps = capture.get(cv.CAP_PROP_FPS)
    width, height = capture.get(cv.CAP_PROP_FRAME_WIDTH), capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    step = max(1, int(fps / SAMPLES_PER_SEC))
    times, boxes = [], []
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0:
                result = detector.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                box = result.detections[0].location_data.relative_bounding_box if result.detections else None
                times.append(frame_index / fps)
                boxes.append((np.nan,) * 3 if box is None else
                             ((box.xmin + box.width / 2) * width, (box.ymin + box.height / 2) * height, box.height * height))
            frame_index += 1
    capture.release()
    return np.asarray(times), np.asarray(boxes)


def head_motion(times, boxes):
    centre, face_height = boxes[:, :2], np.nanmedian(boxes[:, 2])
    found = np.isfinite(centre[:, 0])
    spread = np.nanpercentile(centre, 95, axis=0) - np.nanpercentile(centre, 5, axis=0)

    filled = np.stack([np.interp(times, times[found], centre[found, k]) for k in range(2)], axis=1)
    rate = 1 / np.median(np.diff(times))
    width = max(1, int(round(rate)))
    smoothed = np.stack([np.convolve(filled[:, k], np.ones(width) / width, mode="valid") for k in range(2)], axis=1)
    speed = np.hypot(*np.diff(smoothed, axis=0).T).mean() * rate

    return 100 * np.hypot(*spread) / face_height, 100 * speed / face_height, 1 - found.mean(), face_height


def run(trial, track_dir):
    times, boxes = face_track(video_path(trial))
    np.savez(os.path.join(track_dir, f"{trial}.npz"), time_sec=times, centre_x_px=boxes[:, 0], centre_y_px=boxes[:, 1],
             face_height_px=boxes[:, 2])
    return (trial, *head_motion(times, boxes))


if __name__ == "__main__":
    track_dir, jobs = sys.argv[1], int(sys.argv[2])
    os.makedirs(track_dir, exist_ok=True)
    trials = [t for prefix in ("cohface", "mahnob_w2")
              for t in pd.read_csv(os.path.join(HERE, "rate_agreement", f"{prefix}_motion_trials.csv")).trial.unique()]
    out = csv.writer(sys.stdout)
    out.writerow(["trial", "range_pct_face", "speed_pct_face_per_sec", "miss_share", "face_height_px"])
    with Pool(jobs) as pool:
        for row in pool.starmap(run, [(t, track_dir) for t in trials]):
            out.writerow([row[0]] + [f"{v:.4f}" for v in row[1:]])
