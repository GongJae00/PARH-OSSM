#!/usr/bin/env bash
# The tests on the MAHNOB-HCI test subjects: the confirmatory comparison of the frozen PARH-OSSM with the
# submitted procedure (run w2), then the secondary comparisons against the reliability controls, the component
# ladder, and the leave-one-out runs, once per belt placement. This computes test-subject errors, so run it only
# when the placement is settled. It reads the CSVs of evaluation/window_rates.sh and evaluation/bank_rates.sh.
#
# usage (from the repository root): evaluation/mahnob_test_scores.sh <records dir> <out dir>
set -e
records=$1; out=$2
split=evaluation/mahnob_subject_split.csv
controls=(equal_readout shuffled_readout equal_filter)
rungs=(plain_multichannel readout observation_law observability)
exclusions=(no_torso no_of no_dof no_p1d no_bridges no_p1d_cubic no_p1d_consensus)
mkdir -p "$out"

for placement in onset tail; do
  suffix=$([ $placement = onset ] && echo _onset || echo "")
  frozen_csv=$records/mahnob_frozen${suffix}_window_rates_three_references.csv
  python evaluation/confirmatory_test.py "$frozen_csv" "$records/mahnob_w2${suffix}_window_rates_three_references.csv" "$split" test \
    > "$out/mahnob_confirmatory_test_${placement}.txt"
  python evaluation/reliability_controls_summary.py "$frozen_csv" "$split" test \
    equal_weight="$records/mahnob_equal_weight_track_rates.csv:equal_weight_track_bpm" \
    $(for c in "${controls[@]}"; do echo "$c=$records/mahnob_${c}${suffix}_window_rates_three_references.csv:parh_track_bpm"; done) \
    > "$out/mahnob_reliability_controls_test_${placement}.txt"
  python evaluation/reliability_controls_summary.py "$frozen_csv" "$split" test \
    ossm_kf="$frozen_csv:ossm_kf_track_bpm" median_fusion="$records/mahnob_observation_fusion_window_rates.csv:median_fusion_nine_bpm" \
    $(for r in "${rungs[@]}"; do echo "$r=$records/mahnob_${r}${suffix}_window_rates_three_references.csv:parh_track_bpm"; done) \
    > "$out/mahnob_component_ladder_test_${placement}.txt"
  python evaluation/reliability_controls_summary.py "$frozen_csv" "$split" test \
    $(for e in "${exclusions[@]}"; do echo "$e=$records/mahnob_loo9_${e}${suffix}_window_rates_three_references.csv:parh_track_bpm"; done) \
    > "$out/mahnob_leave_one_out_test_${placement}.txt"
done
