"""Extract the observation signals of a dataset and filter each one with OSSM-KF and PARH-OSSM.

usage (from the repository root):
  python observations/extract.py --config configs/<config>.json --dataset-root <dataset folder> --results <results folder>

<dataset folder> is the COHFACE root (<subject>/<trial>/data.cfr.avi, data.hdf5) or the MAHNOB-HCI root
(<session>/*.avi, *.bdf) named by the dataset entry of the configuration. Observation signals and the ROI
statistics are cached beside each video (obs_*.npy, obs_roi_stats_v1.npz), so the folder must be writable;
observations/run_shards.py mirrors read-only recordings into such a folder. The per-trial result is written
to <results folder>/<config name>/data/<dataset>_<trial>.pkl.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from collections import Counter

from estimate.filters.observation_head import create_wrapped_method
from observations.config import load_config
from observations.datasets import COHFACE, MAHNOB
from observations.signals.observation_signals import (
    DoF_Model,
    DoFDisplacementBridge_Model,
    OF_Model,
    OFDisplacementBridge_Model,
    Profile1DConsensus_Model,
    TorsoShift_Model,
    profile1D_Model,
)
from observations.trials.trial_runner import extract_respiration

# Execution order of the observation families in the stored estimates.
FAMILY_ORDER = {
    'of_farneback': 10, 'of_disp_bridge': 15, 'dof': 20, 'dof_disp_bridge': 25,
    'profile1d_linear': 30, 'profile1d_quadratic': 40, 'profile1d_cubic': 50, 'profile1d_consensus': 57,
}


def _method_order_key(name):
    normalized = name.strip().lower().replace(' ', '_')
    base, _, head = normalized.partition('__')
    head_rank = 0 if not head else (10 if 'kfstd' in head else 90)
    return FAMILY_ORDER.get(base, 99), head_rank, normalized


def _build_datasets(dataset_configs, dataset_root):
    constructors = {'cohface': COHFACE, 'mahnob': MAHNOB}
    datasets = []
    for dataset_config in dataset_configs:
        dataset = constructors[dataset_config['name'].lower()](dataset_root)
        dataset.configure(dataset_config)
        datasets.append(dataset)
    return datasets


def _build_methods(method_configs, config):
    methods = []
    for entry in method_configs:
        name = entry['name']
        if '__' in name:
            methods.append(create_wrapped_method(
                name,
                params=entry,
                oscillator_defaults=config.get('oscillator', {}),
                preproc_defaults=config.get('preproc', {}),
                gating_defaults=config.get('gating', {}),
                quality_defaults=config.get('quality', {}),
                trust_defaults=config.get('trust', {}),
                gating_scope_default=config.get('gating_scope', 'evaluation_only'),
            ))
        elif name == 'of_farneback':
            methods.append(OF_Model())
        elif name == 'of_disp_bridge':
            methods.append(OFDisplacementBridge_Model())
        elif name == 'dof':
            methods.append(DoF_Model())
        elif name == 'dof_disp_bridge':
            methods.append(DoFDisplacementBridge_Model())
        elif name == 'profile1d_consensus':
            methods.append(Profile1DConsensus_Model())
        elif name.startswith('profile1d_'):
            methods.append(profile1D_Model(name.split('_')[1]))
        elif name == 'torso_shift':
            methods.append(TorsoShift_Model())
        else:
            print(f"Warning: Unknown method {name}")
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', '-c', required=True, help='Path to config JSON')
    parser.add_argument('--dataset-root', required=True, help='Dataset folder holding the trials')
    parser.add_argument('--results', '-r', required=True, help='Results directory')
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = _build_datasets(config['datasets'], args.dataset_root)
    methods = sorted(_build_methods(config['methods'], config), key=lambda m: _method_order_key(m.name))
    duplicates = [name for name, count in Counter(m.name for m in methods).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate method names in config: {duplicates}")
    print(f"Loaded {len(datasets)} datasets and {len(methods)} methods.")

    extract_respiration(
        datasets=datasets,
        methods=methods,
        results_dir=args.results,
        run_label=config.get('name'),
    )


if __name__ == "__main__":
    main()
