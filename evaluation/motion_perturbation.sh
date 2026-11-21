#!/usr/bin/env bash
# The frozen PARH-OSSM on COHFACE videos carrying a controlled non-respiratory perturbation
# (evaluation/motion_perturbation.py), one trial per subject (evaluation/cohface_perturbation_trials.txt, 40 trials).
# The baseline is the unperturbed from-video run with the chest ROI fixed at the first frame, so a lossless copy
# without perturbation (level 0) is frame-identical to the source. Every perturbed copy goes through the same
# path and the same config: extraction of the nine observations from the video, the bank
# (estimate/drop_observations.py bank), then the materializer with the frozen flags and the reliability priors fitted
# on the unperturbed videos' bank, so only the input changes. The perturbed videos are deleted once their
# observations are extracted.
#
# usage (from the repository root):
#   evaluation/motion_perturbation.sh <COHFACE folder> <priors csv of the unperturbed bank> <work dir> <records dir> <jobs> <kind>:<level> ...
set -e
source=$1; priors=$2; root=$3; records=$4; jobs=$5; shift 5
config=configs/cohface_first_frame_roi.json
trials=($(cat evaluation/cohface_perturbation_trials.txt))
mkdir -p "$records"

for condition in "$@"; do
  kind=${condition%%:*}; level=${condition##*:}
  work=$root/${kind}_${level}
  printf '%s\n' "${trials[@]}" | xargs -n 10 -P "$jobs" python evaluation/motion_perturbation.py \
    "$source" "$work/video" "$kind" "$level"
  python observations/run_shards.py --dataset COHFACE --dataset-root "$work/video/COHFACE" --config "$config" \
    --work "$work" --shards "$jobs" --no-caches
  python estimate/drop_observations.py "$work/data" "$work/bank/data" bank
  estimate/materialize.sh frozen "$work/bank/data" "$priors" "$work/bank" "$jobs"
  python evaluation/window_rates_three_references.py "$work/data" "$work/bank/final/data" "$jobs" \
    > "$records/cohface_perturbation_${kind}_${level}_window_rates_three_references.csv"
  find "$work/video" -name data.cfr.avi -type f -delete
  echo "$condition done"
done
