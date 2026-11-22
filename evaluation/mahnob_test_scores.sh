#!/usr/bin/env bash
# The scoring of the MAHNOB-HCI participants of one split and the Supplementary tables of both datasets. With split
# test it computes the errors of the test participants, so run it only when the placement is settled; with split
# development it runs the same chain on the development participants, which shows a missing file or a crash before any
# test score exists. It writes the rows of Table 2 (main_tables.sh; for test also the development and
# all-trial rows), the confirmatory comparison of the frozen PARH-OSSM with the submitted procedure (run w2), the
# reliability controls, the component ladder and the route exclusions, each once per belt placement, the waveform agreement under
# the onset placement with its rows of the waveform table, the correlation of the errors of the observation pairs (on all
# 488 trials in both modes, since it reads no reference), the
# frozen run's trial means and head-motion tertiles (onset placement), the slopes of the trial means by stage and of the
# observation tracks, and the Supplementary tables. The scripts that read the split from <records dir> get a copy of it there.
# <base data dir> holds the pickles of observations/extract.py (the nine observations of all 488 common trials),
# <bank data dir> those of the bank (estimate/bank_priors.sh), and <final data dir> those of the frozen run of
# estimate/materialize.sh. The Supplementary tables also read the outputs of evaluation/cohface_scores.sh and
# head_motion_trials.csv of evaluation/motion/head_motion.py, all in <records dir>. With split development, the waveform
# scoring reads only the development trials, which it links into <records dir>/scoring_development.
#
# usage (from the repository root):
#   evaluation/mahnob_test_scores.sh <records dir> <out dir> <base data dir> <bank data dir> <final data dir> \
#       <MAHNOB-HCI folder> <jobs> [split: test|development, default test]
set -e
records=$1; out=$2; base=$3; bank=$4; final=$5; mahnob=$6; jobs=$7; evaluated=${8:-test}
split=configs/evaluation/mahnob_subject_split.csv
controls=(equal_readout shuffled_readout equal_filter)
rungs=(plain_multichannel readout observation_law observability)
exclusions=($(python estimate/bank/bank.py routes))
splits=$([ "$evaluated" = test ] && echo "test development all" || echo development)
mkdir -p "$out" "$records/rate_agreement"
cp "$split" "$records/"

# the breathing-band coherence reads no reference, so it is computed on all of $base in both modes
all_base=$base
if [ "$evaluated" = development ]; then
  scoring=$records/scoring_development
  mkdir -p "$scoring/data" "$scoring/final"
  for trial in $(awk -F, 'NR > 1 && $0 ~ /development/ {print $1}' "$split"); do
    [ -f "$final/$trial.pkl" ] || continue  # two development sessions of the split are not among the 488 common sessions
    ln -sf "$(realpath "$base/$trial.pkl")" "$scoring/data/"; ln -sf "$(realpath "$final/$trial.pkl")" "$scoring/final/"
  done
  base=$scoring/data; final=$scoring/final
fi

python evaluation/waveform/onset_alignment_waveform.py "$base" "$final" "$mahnob" "$jobs" > "$records/mahnob_${evaluated}_alignment_waveform.csv"
python evaluation/waveform/waveform_table.py "$records" mahnob "$out/T_waveform_mahnob.tex" "$evaluated" > "$out/mahnob_${evaluated}_waveform_table.txt"
python evaluation/reliability/observation_error_correlation.py "$all_base" "$records/mahnob_error_correlation.csv" - "$jobs" \
  > "$out/mahnob_error_correlation.txt"

for placement in onset tail; do
  suffix=$([ $placement = onset ] && echo _onset || echo "")
  frozen_csv=$records/mahnob_frozen${suffix}_window_rates_three_references.csv
  python evaluation/rates/confirmatory_test.py "$frozen_csv" "$records/mahnob_w2${suffix}_window_rates_three_references.csv" "$split" "$evaluated" \
    > "$out/mahnob_confirmatory_${evaluated}_${placement}.txt"
  python evaluation/reliability/reliability_controls_summary.py "$frozen_csv" "$split" "$evaluated" \
    equal_weight="$records/mahnob_equal_weight_track_rates.csv:equal_weight_track_bpm" \
    $(for c in "${controls[@]}"; do echo "$c=$records/mahnob_${c}${suffix}_window_rates_three_references.csv:parh_track_bpm"; done) \
    > "$out/mahnob_reliability_controls_${evaluated}_${placement}.txt"
  python evaluation/reliability/reliability_controls_summary.py "$frozen_csv" "$split" "$evaluated" \
    ossm_kf="$frozen_csv:ossm_kf_track_bpm" median_fusion="$records/mahnob_observation_fusion_window_rates.csv:median_fusion_bank_bpm" \
    $(for r in "${rungs[@]}"; do echo "$r=$records/mahnob_${r}${suffix}_window_rates_three_references.csv:parh_track_bpm"; done) \
    > "$out/mahnob_component_ladder_${evaluated}_${placement}.txt"
  python evaluation/reliability/reliability_controls_summary.py "$frozen_csv" "$split" "$evaluated" \
    $(for e in "${exclusions[@]}"; do echo "$e=$records/mahnob_${e}${suffix}_window_rates_three_references.csv:parh_track_bpm"; done) \
    > "$out/mahnob_route_exclusion_${evaluated}_${placement}.txt"
done

evaluation/main_tables.sh "$records" mahnob "$splits" > "$records/mahnob_${evaluated}_main_tables.tex"

python evaluation/across_trials/rate_agreement.py MAHNOB-HCI "$records/mahnob_frozen_onset_window_rates_three_references.csv" \
  "$records/mahnob_observation_fusion_window_rates.csv:median_fusion_bank_bpm" "$records/rate_agreement/mahnob_frozen_onset_trial_means.csv" \
  "$split" "$evaluated" > "$out/mahnob_frozen_onset_rate_agreement_${evaluated}.txt"
python evaluation/motion/motion_tertiles.py MAHNOB-HCI "$records/mahnob_frozen_onset_window_rates_three_references.csv" \
  "$records/head_motion_trials.csv" "$records/rate_agreement/mahnob_frozen_onset_head_motion_trials.csv" "$split" "$evaluated" \
  > "$out/mahnob_frozen_onset_head_motion_tertiles_${evaluated}.txt"
for reference in hilbert welch count; do
  python evaluation/across_trials/stage_slopes.py "$records" $reference mahnob "$evaluated"
done > "$out/mahnob_stage_slopes_${evaluated}.txt"
python evaluation/across_trials/observation_track_slope.py "$records" "$bank" "$final" "$evaluated" "$jobs" \
  > "$out/mahnob_observation_track_slope_${evaluated}.txt"

python evaluation/tables/supplementary_tables.py "$records" "$out" "$evaluated" "$(IFS=,; echo "${exclusions[*]}")"
for reference in hilbert welch count; do
  python figures/rate_agreement_figure.py "$records" $reference "$out/S_rate_agreement_$reference.pdf" "$evaluated"
done
