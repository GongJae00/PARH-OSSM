#!/usr/bin/env bash
# Run the multichannel PARH-OSSM filter on a folder of observation pickles with the frozen flags, or with
# one thing changed: a rung of the component ladder or a control.
#
#   frozen              the reported PARH-OSSM (the flags below)
#   plain_multichannel  rung 1: the multichannel filter alone, the rate is its own frequency track
#   readout             rung 2: + the decoupled readout of the observations' OSSM-KF tracks
#   observation_law     rung 3: + the observation law inside the filter (keeps the readout on)
#   observability       rung 4: + target observability control and signal-quality observability
#   equal_readout       control: readout priors equal across observations in each window
#   shuffled_readout    control: each observation's prior windows permuted within the trial
#   equal_filter        control: the filter's per-observation trust replaced by its mean over observations
#   no_of no_dof no_p1d no_bridges no_p1d_cubic no_p1d_consensus no_torso
#                       leave-one-out: the frozen method on a copy of the pickles without those observations
#                       (the optional sessions file restricts a MAHNOB-HCI copy to the sessions listed)
#
# usage (from the repository root):
#   estimate/materialize.sh <variant> <data dir> <priors csv> <out dir> <jobs> [sessions file]
# <data dir> holds the pickles of observations/extract.py, <priors csv> is the group file of
# estimate/reliability_priors.py. The pickles of the run are written to <out dir>/final/data.
set -e
variant=$1; data=$2; priors=$3; out=$4; jobs=$5; sessions=$6
script=estimate/multichannel_fusion.py
flags=(--enable-observation-law --enable-rate-posterior --enable-target-observability-control
       --enable-signal-sqi-observability --rate-posterior-output-source final)
mkdir -p "$out"

case $variant in
  frozen) ;;
  plain_multichannel) flags=() ;;
  readout) flags=(--enable-decoupled-rate-readout) ;;
  observation_law) flags=(--enable-observation-law) ;;
  observability) flags=(--enable-observation-law --enable-target-observability-control --enable-signal-sqi-observability) ;;
  equal_readout|shuffled_readout)
    python estimate/reliability_controls.py "$priors" "${variant%_readout}" "$out/priors_group.csv"
    priors=$out/priors_group.csv ;;
  equal_filter) script=estimate/equal_filter_trust.py ;;
  no_*)
    python estimate/leave_one_out_data.py "$data" "$out/data" "$variant" $sessions
    data=$out/data ;;
esac

python "$script" --data-dir "$data" --out-run "$out/final" --name parh_ossm --parh-input multichannel \
  --external-rate-evidence-source ossm_kfstd "${flags[@]}" --eval-use-track --jobs "$jobs" \
  --reliability-group-csv "$priors" --reliability-prior-scope readout_only
