#!/usr/bin/env bash
# The observation bank and its reliability priors: the pickles of observations/extract.py with the
# observations the bank does not use removed (estimate/drop_observations.py bank), and the GT-free reliability priors
# fitted on the bank itself in 30 s windows every 10 s (estimate/reliability_priors.py). The priors read no belt reference
# and hold one set of rows per video, so fitting them on the 178 development trials or on all 488 MAHNOB-HCI trials
# gives the same rows for a development video.
#
# usage (from the repository root):
#   estimate/bank_priors.sh <dataset label: COHFACE|MAHNOB_tailaligned> <data dir of extract.py> <out dir> <jobs> [variant, default bank]
# The variant is an added-back run of estimate/bank.py (with_dof), whose bank and priors the route table needs.
# Writes <out dir>/data (the bank), <out dir>/priors_group.csv and <out dir>/priors_summary.csv.
set -e
label=$1; data=$2; out=$3; jobs=$4; variant=${5:-bank}
mkdir -p "$out"
python estimate/drop_observations.py "$data" "$out/data" "$variant"
python estimate/reliability_priors.py --data-dir "$out/data" --dataset-label "$label" \
  --out-edge "$out/priors_edges.csv" --out-group "$out/priors_group.csv" --out-summary "$out/priors_summary.csv" \
  --window-sec 30 --window-stride-sec 10 --jobs "$jobs"
rm "$out/priors_edges.csv"
