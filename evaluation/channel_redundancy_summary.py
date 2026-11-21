"""Summarise channel_redundancy.py output: median over trials per pair, and within- vs between-family.

usage: channel_redundancy_summary.py <dataset>_channel_redundancy.csv
"""
import sys

import pandas as pd

FAMILY = {"of_farneback": "OF", "of_disp_bridge": "OF", "DoF": "DoF", "dof_disp_bridge": "DoF",
          "profile1D linear": "P1D", "profile1D quadratic": "P1D", "profile1D cubic": "P1D", "profile1d_consensus": "P1D",
          "torso_shift": "Torso"}

pairs = pd.read_csv(sys.argv[1])
pairs["relation"] = [
    f"within {FAMILY[a]}" if FAMILY[a] == FAMILY[b] else "between " + "-".join(sorted((FAMILY[a], FAMILY[b])))
    for a, b in zip(pairs.channel_a, pairs.channel_b)]
print(f"{pairs.trial.nunique()} trials")

per_pair = pairs.groupby(["channel_a", "channel_b", "relation"]).agg(
    waveform_similarity=("waveform_similarity", "median"),
    rate_gap_bpm=("rate_gap_bpm", "median"),
    share_rate_gap_under_1=("rate_gap_bpm", lambda g: (g < 1).mean()),
    flat_trials=("waveform_similarity", lambda s: s.isna().sum()),
).reset_index()
print(per_pair.round(3).to_string(index=False))
print()
print(pairs.groupby("relation").agg(
    pairs_x_trials=("rate_gap_bpm", "size"),
    waveform_similarity=("waveform_similarity", "median"),
    rate_gap_bpm=("rate_gap_bpm", "median"),
    share_rate_gap_under_1=("rate_gap_bpm", lambda g: (g < 1).mean()),
).round(3).to_string())
