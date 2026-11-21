#!/usr/bin/env bash
# Run the multichannel PARH-OSSM filter on a folder of observation pickles with the frozen flags, or with
# one thing changed: a rung of the component ladder, a control, or a measurement route left out of the bank.
#
#   frozen              the reported PARH-OSSM (the flags below)
#   plain_multichannel  rung 1: the multichannel filter alone, the rate is its own frequency track
#   readout             rung 2: + the decoupled readout of the observations' OSSM-KF tracks
#   observation_law     rung 3: + the observation law inside the filter (keeps the readout on)
#   observability       rung 4: + target observability control and signal-quality observability
#   equal_readout       control: readout priors equal across observations in each window
#   shuffled_readout    control: each observation's prior windows permuted within the trial
#   equal_filter        control: the filter's per-observation trust replaced by its mean over observations
#   no_of no_dof no_p1d no_torso
#                       route exclusion: the frozen method on a copy of the bank's pickles without that
#                       measurement route (no_of: OF; no_dof: DoF and the DoF bridge; no_p1d: P1D quadratic;
#                       no_torso: torso displacement), with the priors of the full bank
#
# usage (from the repository root):
#   estimate/materialize.sh <variant> <data dir> <priors csv> <out dir> <jobs>
# <data dir> holds the pickles of the bank (estimate/drop_observations.py bank applied to the pickles of
# observations/extract.py), <priors csv> is the group file of estimate/reliability_priors.py computed on them.
# The pickles of the run are written to <out dir>/final/data.
set -e
variant=$1; data=$2; priors=$3; out=$4; jobs=$5
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
  no_of|no_dof|no_p1d|no_torso)
    python estimate/drop_observations.py "$data" "$out/data" "$variant"
    data=$out/data ;;
  *) echo "unknown run: $variant" >&2; exit 1 ;;
esac

python "$script" --data-dir "$data" --out-run "$out/final" --name parh_ossm --parh-input multichannel \
  --external-rate-evidence-source ossm_kfstd "${flags[@]}" --eval-use-track --jobs "$jobs" \
  --reliability-group-csv "$priors" --reliability-prior-scope readout_only
