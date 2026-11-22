#!/usr/bin/env bash
# Window rates of PARH-OSSM runs against the three belt references (Hilbert, Welch, breath count), one CSV per run.
# On MAHNOB-HCI the belt is read under both placements: the tail placement (the stored pickles) and the onset
# placement (evaluation/reference/onset_reference.py re-aligns each pickle's reference to the start of the recording).
# Writes per-window rates only, no trial error or test.
#
# usage (from the repository root):
#   evaluation/window_rates.sh <dataset: cohface|mahnob> <base data dir> <runs dir> <records dir> <jobs> <MAHNOB-HCI folder | -> <run> ...
# <base data dir> holds the pickles of observations/extract.py (all nine observations, for direct P1D), and <runs dir>/<run>/final/data those of
# estimate/materialize.sh. Each run gives <records dir>/<dataset>_<name>_window_rates_three_references.csv, plus
# <dataset>_<name>_onset_window_rates_three_references.csv when a MAHNOB-HCI folder is given; <name> is the run's name.
set -e
dataset=$1; base=$2; runs=$3; records=$4; jobs=$5; mahnob=$6; shift 6
mkdir -p "$records"
for run in "$@"; do
  name=$run
  python evaluation/rates/window_rates_three_references.py "$base" "$runs/$run/final/data" "$jobs" \
    > "$records/${dataset}_${name}_window_rates_three_references.csv"
  if [ "$mahnob" != - ]; then
    python evaluation/reference/onset_reference.py "$runs/$run/final/data" "$runs/$run/onset_final/data" "$mahnob"
    python evaluation/rates/window_rates_three_references.py "$base" "$runs/$run/onset_final/data" "$jobs" \
      > "$records/${dataset}_${name}_onset_window_rates_three_references.csv"
  fi
done
