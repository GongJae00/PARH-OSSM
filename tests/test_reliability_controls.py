import os
import subprocess
import sys

import numpy as np
import pandas as pd

ESTIMATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "estimate")


def priors_file(tmp_path):
    rows = [{"video": video, "group": group, "window_id": window, "window_start_sec": 10 * window,
             "window_end_sec": 10 * window + 30, "window_duration_sec": 30, "score": 10 * g + window}
            for video in ("a", "b") for g, group in enumerate(("G_OF", "G_DoF", "G_P1D")) for window in range(4)]
    path = tmp_path / "priors.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_control(tmp_path, control):
    out = tmp_path / f"{control}.csv"
    subprocess.run([sys.executable, os.path.join(ESTIMATE, "controls", "reliability_controls.py"), str(priors_file(tmp_path)), control,
                    str(out)], check=True)
    return pd.read_csv(out)


def test_equal_control_gives_every_group_the_window_mean(tmp_path):
    controlled = run_control(tmp_path, "equal")

    assert (controlled.groupby(["video", "window_id"]).score.nunique() == 1).all()
    assert np.allclose(controlled.groupby("window_id").score.mean(), [10, 11, 12, 13])


def test_shuffled_control_keeps_each_groups_scores_within_its_trial(tmp_path):
    original = pd.read_csv(priors_file(tmp_path))

    controlled = run_control(tmp_path, "shuffled")

    for (video, group), rows in original.groupby(["video", "group"]):
        shuffled = controlled[(controlled.video == video) & (controlled.group == group)]
        assert sorted(shuffled.score) == sorted(rows.score)
