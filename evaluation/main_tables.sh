#!/usr/bin/env bash
# Rows of the manuscript's Table 2 (methods on the same observations) and Table 3 (the component ladder), one
# invocation of main_tables.py per dataset and split. The records are the CSVs of evaluation/window_rates.sh and
# evaluation/bank_rates.sh, named <dataset>_<run>[_onset]_window_rates_three_references.csv.
#   cohface: all 160 trials, PARH-OSSM on the first-frame observations; the row "as submitted" is the submitted
#     procedure on them: run chest_only, the eight chest observations with priors computed from them
#   mahnob:  the primary onset placement, then the tail placement, each for the splits listed (test, development,
#     all 488 common trials); the row "as submitted" is run w2, the frozen flags with the submitted priors on the eight
#     chest observations of the submitted extraction
# usage (from the repository root): evaluation/main_tables.sh <records dir> cohface > <rows>.tex
#                                   evaluation/main_tables.sh <records dir> mahnob "<splits>" > <rows>.tex
set -e
records=$1
run=(python evaluation/main_tables.py)
split=evaluation/mahnob_subject_split.csv

# table <dataset label> <split csv | none> <evaluated split> <dataset> <placement suffix: "" or _onset> <as-submitted run>
table() {
  local label=$1 split_csv=$2 evaluated=$3 dataset=$4 suffix=$5 submitted=$6
  local frozen=$records/${dataset}_frozen${suffix}_window_rates_three_references.csv
  local fusion=$records/${dataset}_observation_fusion_window_rates.csv
  local stage=$records/${dataset}_
  local end=${suffix}_window_rates_three_references.csv
  echo "% Table 2"
  "${run[@]}" "$label" "$split_csv" "$evaluated" \
    "Direct P1D quadratic|signal spectrum=$frozen:direct_p1d_welch_bpm" \
    "Direct torso displacement|signal spectrum=$fusion:direct_torso_bpm" \
    "Median fusion|signal spectra=$fusion:median_fusion_bank_bpm" \
    "OSSM-KF|frequency track=$frozen:ossm_kf_track_bpm" \
    "PARH-OSSM as submitted|posterior-refined track=$stage${submitted}$end:parh_track_bpm" \
    "PARH-OSSM|posterior-refined track=$frozen:parh_track_bpm"
  echo "% Table 3"
  "${run[@]}" "$label" "$split_csv" "$evaluated" \
    "Single-observation OSSM-KF|frequency track=$frozen:ossm_kf_track_bpm" \
    "Median fusion|signal spectra=$fusion:median_fusion_bank_bpm" \
    "Multichannel PARH filter|filter frequency track=${stage}plain_multichannel$end:parh_track_bpm" \
    "+ readout of observation tracks|readout=${stage}readout$end:parh_track_bpm" \
    "+ observation law|readout=${stage}observation_law$end:parh_track_bpm" \
    "+ target observability and signal quality|readout=${stage}observability$end:parh_track_bpm" \
    "+ rate posterior (PARH-OSSM)|posterior-refined track=$frozen:parh_track_bpm"
}

case $2 in
cohface)
  table COHFACE none all cohface "" chest_only ;;
mahnob)
  for placement in onset tail; do
    suffix=$([ $placement = onset ] && echo _onset || echo "")
    for evaluated in $3; do
      echo "% placement $placement"
      table MAHNOB-HCI "$split" $evaluated mahnob "$suffix" w2
    done
  done ;;
esac
