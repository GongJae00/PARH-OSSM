#!/usr/bin/env bash
# The COHFACE analyses behind the waveform table and the Supplementary, on all 160 trials. COHFACE was used to build
# the method, so these are descriptive and none of them is a test. It writes the waveform agreement of the frozen run
# with the phase-randomized null and the rows of the waveform table, the waveform similarity and rate agreement of
# the observation pairs, the frozen run's trial means and head-motion tertiles, the slopes of the trial means by
# stage, and the per-trial errors on the perturbed copies of evaluation/motion_perturbation.sh. It reads the CSVs of
# evaluation/window_rates.sh cohface, evaluation/bank_rates.sh cohface, and evaluation/motion_perturbation.sh, and
# head_motion_trials.csv of evaluation/head_motion.py, all in <records dir>. evaluation/mahnob_test_scores.sh then
# writes the Supplementary tables of both datasets.
# <base data dir> holds the pickles of observations/extract.py, <final data dir> those of the frozen run of
# estimate/materialize.sh.
#
# usage (from the repository root):
#   evaluation/cohface_scores.sh <records dir> <out dir> <base data dir> <final data dir> <jobs>
set -e
records=$1; out=$2; base=$3; final=$4; jobs=$5
frozen_csv=$records/cohface_frozen_window_rates_three_references.csv
perturbations=(translation:0.5 translation:1 translation:2 rotation:0.25 rotation:0.5 rotation:1
               brightness:0.01 brightness:0.02 brightness:0.05)
mkdir -p "$out" "$records/rate_agreement"

python evaluation/bounded_lag_waveform.py "$base" "$final" - "$jobs" > "$records/cohface_bounded_lag_waveform.csv"
python evaluation/phase_randomized_null.py "$base" "$final" "$jobs" > "$records/cohface_phase_randomized_null.csv"
python evaluation/waveform_table.py "$records" cohface "$out/T_waveform_cohface.tex" > "$out/cohface_waveform_table.txt"
python evaluation/channel_redundancy.py "$base" "$jobs" > "$records/cohface_channel_redundancy.csv"

python evaluation/rate_agreement.py COHFACE "$frozen_csv" \
  "$records/cohface_observation_fusion_window_rates.csv:median_fusion_nine_bpm" \
  "$records/rate_agreement/cohface_frozen_trial_means.csv" > "$out/cohface_frozen_rate_agreement.txt"
python evaluation/motion_tertiles.py COHFACE "$frozen_csv" "$records/head_motion_trials.csv" \
  "$records/rate_agreement/cohface_frozen_head_motion_trials.csv" > "$out/cohface_frozen_head_motion_tertiles.txt"
for reference in hilbert welch count; do
  python evaluation/stage_slopes.py "$records" $reference cohface
done > "$out/cohface_stage_slopes.txt"
python evaluation/motion_perturbation_summary.py "$frozen_csv" evaluation/cohface_perturbation_trials.txt \
  "$records/cohface_perturbation_trial_mae.csv" \
  $(for c in "${perturbations[@]}"; do
      echo "$c=$records/cohface_perturbation_${c%%:*}_${c##*:}_window_rates_three_references.csv"
    done) > "$out/cohface_perturbation_summary.txt"
