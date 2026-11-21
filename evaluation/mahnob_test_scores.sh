#!/usr/bin/env bash
# The tests on the MAHNOB-HCI test subjects: the confirmatory comparison of the frozen PARH-OSSM with the
# submitted procedure (run w2), then the secondary comparisons against the reliability controls, the component
# ladder, and the leave-one-out runs, once per belt placement. This computes test-subject errors, so run it only
# when the placement is settled. It reads the CSVs of evaluation/window_rates.sh and evaluation/bank_rates.sh.
# It then writes the waveform agreement of the frozen run under both placements with the phase-randomized null and the
# rows of the waveform table, the frozen run's trial means on all 488 trials and its head-motion tertiles on the test
# trials (onset placement), the slopes of the trial means by stage and of the observation tracks, the redundancy of
# the nine observations, and the Supplementary tables. The scripts that read the split from <records dir> get a copy of it there.
# <base data dir> holds the pickles of observations/extract.py (the eight chest observations), <bank data dir> those
# of the nine observations (estimate/merge_torso_rows.py), and <final data dir> those of the frozen run of
# estimate/materialize.sh. The Supplementary tables also read the outputs of evaluation/cohface_scores.sh and the
# rows of Tables 2 and 3, <records dir>/cohface_main_tables.tex and mahnob_main_tables.tex (evaluation/main_tables.sh).
#
# usage (from the repository root):
#   evaluation/mahnob_test_scores.sh <records dir> <out dir> <base data dir> <bank data dir> <final data dir> \
#       <MAHNOB-HCI folder> <jobs>
set -e
records=$1; out=$2; base=$3; bank=$4; final=$5; mahnob=$6; jobs=$7
split=evaluation/mahnob_subject_split.csv
controls=(equal_readout shuffled_readout equal_filter)
rungs=(plain_multichannel readout observation_law observability)
exclusions=(no_torso no_of no_dof no_p1d no_bridges no_p1d_cubic no_p1d_consensus)
mkdir -p "$out"
cp "$split" "$records/"

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

python evaluation/onset_alignment_waveform.py "$base" "$final" "$mahnob" "$jobs" > "$records/mahnob_onset_alignment_waveform.csv"
python evaluation/waveform_table.py "$records" mahnob "$out/T_waveform_mahnob.tex" > "$out/mahnob_test_waveform_table.txt"

mkdir -p "$records/rate_agreement"
python evaluation/rate_agreement.py MAHNOB-HCI "$records/mahnob_frozen_onset_window_rates_three_references.csv" \
  "$records/mahnob_observation_fusion_window_rates.csv:median_fusion_nine_bpm" "$records/rate_agreement/mahnob_frozen_onset_trial_means.csv" \
  "$split" > "$out/mahnob_frozen_onset_rate_agreement.txt"
python evaluation/motion_tertiles.py MAHNOB-HCI "$records/mahnob_frozen_onset_window_rates_three_references.csv" \
  "$records/head_motion_trials.csv" "$records/rate_agreement/mahnob_frozen_onset_head_motion_trials.csv" "$split" \
  > "$out/mahnob_frozen_onset_head_motion_tertiles.txt"
for reference in hilbert welch count; do
  python evaluation/stage_slopes.py "$records" $reference mahnob test
done > "$out/mahnob_test_stage_slopes.txt"
python evaluation/observation_track_slope.py "$records" "$bank" "$final" test "$jobs" > "$out/mahnob_test_observation_track_slope.txt"
python evaluation/channel_redundancy.py "$bank" "$jobs" > "$records/mahnob_channel_redundancy.csv"
python evaluation/supplementary_tables.py "$records" "$out"
