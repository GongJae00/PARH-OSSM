#!/usr/bin/env bash
# Rows of the manuscript's Table 2 (methods on the same nine observations) and Table 3 (the component ladder), one
# invocation of main_tables.py per dataset and split. The records are the CSVs of evaluation/window_rates.sh and
# evaluation/bank_rates.sh, named <dataset>_<run>[_onset]_window_rates_three_references.csv.
#   cohface: all 160 trials, the frozen run on the first-frame observations; the chest-only row is the submitted
#     procedure on them: run chest_only is the no_torso copy with priors computed from it
#   mahnob:  the primary onset placement, then the tail placement, each for the test subjects, the development
#     subjects, and all 488 common trials; the chest-only row is run w2, the frozen flags with the joint
#     COHFACE and MAHNOB-HCI priors on the eight chest observations' bank
# usage (from the repository root): evaluation/main_tables.sh <records dir> <cohface|mahnob> > <rows>.tex
set -e
records=$1
run=(python evaluation/main_tables.py)
split=evaluation/mahnob_subject_split.csv

# table <dataset label> <split csv | none> <evaluated split> <dataset> <placement suffix: "" or _onset> <chest-only run>
table() {
  local label=$1 split_csv=$2 evaluated=$3 dataset=$4 suffix=$5 chest_only=$6
  local frozen=$records/${dataset}_frozen${suffix}_window_rates_three_references.csv
  local fusion=$records/${dataset}_observation_fusion_window_rates.csv
  local stage=$records/${dataset}_
  local end=${suffix}_window_rates_three_references.csv
  echo "% Table 2"
  "${run[@]}" "$label" "$split_csv" "$evaluated" \
    "Direct P1D quadratic|signal spectrum=$frozen:direct_p1d_welch_bpm" \
    "Direct torso displacement|signal spectrum=$fusion:direct_torso_bpm" \
    "Median fusion|signal spectra=$fusion:median_fusion_nine_bpm" \
    "OSSM-KF comparator|frequency track=$frozen:ossm_kf_track_bpm" \
    "PARH-OSSM, chest observations only|posterior-refined track=$stage${chest_only}$end:parh_track_bpm" \
    "PARH-OSSM|posterior-refined track=$frozen:parh_track_bpm"
  echo "% Table 3"
  "${run[@]}" "$label" "$split_csv" "$evaluated" \
    "Single-observation OSSM-KF|frequency track=$frozen:ossm_kf_track_bpm" \
    "Median fusion|signal spectra=$fusion:median_fusion_nine_bpm" \
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
  for suffix in _onset ""; do
    for evaluated in test development all; do
      echo "% placement ${suffix:-tail}"
      table MAHNOB-HCI "$split" $evaluated mahnob "$suffix" w2
    done
  done ;;
esac
