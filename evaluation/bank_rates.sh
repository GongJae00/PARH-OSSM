#!/usr/bin/env bash
# Window rates that read the nine observations of a bank without any filter: the median spectral fusion and the
# direct torso displacement (Table 2), and the unweighted median of the nine rate tracks (the equal-weight control).
#
# usage (from the repository root): evaluation/bank_rates.sh <dataset: cohface|mahnob> <bank data dir> <records dir> <jobs>
# <bank data dir> holds the pickles of the nine observations (observations/extract.py plus the torso rows).
set -e
dataset=$1; bank=$2; records=$3; jobs=$4
mkdir -p "$records"
python evaluation/observation_fusion_window_rates.py "$bank" "$jobs" > "$records/${dataset}_observation_fusion_window_rates.csv"
python evaluation/equal_weight_track_rates.py "$bank" "$jobs" > "$records/${dataset}_equal_weight_track_rates.csv"
