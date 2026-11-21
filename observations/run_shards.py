"""Run observations/extract.py on every trial of a dataset in parallel shards, outside the dataset folder.

usage (from the repository root):
  python observations/run_shards.py --dataset <COHFACE|MAHNOB> --dataset-root <dataset folder>
    --config configs/<config>.json --work <work folder> --shards <n>
    [--trials <file of trial ids>] [--torso-cache <folder>] [--no-caches]

<dataset folder> is read only. Each shard mirrors its trials under <work>/shard_NN/data: recordings are symlinked,
and the observation caches (obs_*) are copied, because extract.py loads them from, and writes new ones into,
the trial folder. --no-caches mirrors only the video and the belt, so every observation is computed from the video.
--torso-cache names a folder of mahnob_<session>.npy torso displacement signals (observations/torso_shift_cache.py),
copied into each mirror as obs_torso_shift.npy; the MAHNOB-HCI extraction takes it, the COHFACE one computes the torso
displacement itself. A trial id is <subject>/<trial> for COHFACE and the session id for
MAHNOB. The pickles of all shards are linked into <work>/data, the folder estimate/ reads.
"""
import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def list_trials(dataset, root):
    if dataset == "COHFACE":
        return [f"{subject}/{trial}"
                for subject in sorted(os.listdir(root), key=lambda x: int(x) if x.isdigit() else -1) if subject.isdigit()
                for trial in sorted(os.listdir(os.path.join(root, subject)))
                if os.path.isdir(os.path.join(root, subject, trial))]
    return [session for session in sorted(os.listdir(root), key=lambda x: int(x) if x.isdigit() else -1)
            if any(name.lower().endswith(".bdf") for name in os.listdir(os.path.join(root, session)))]


def mirror_trial(root, mirror_root, trial, no_caches):
    source = os.path.join(root, trial)
    target = os.path.join(mirror_root, trial)
    os.makedirs(target, exist_ok=True)
    for name in os.listdir(source):
        if os.path.lexists(os.path.join(target, name)):
            continue
        if name.startswith("obs_"):
            if not no_caches:
                shutil.copy2(os.path.join(source, name), os.path.join(target, name))
        else:
            os.symlink(os.path.join(source, name), os.path.join(target, name))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=["COHFACE", "MAHNOB"])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--trials")
    parser.add_argument("--torso-cache")
    parser.add_argument("--no-caches", action="store_true")
    args = parser.parse_args()

    trials = list_trials(args.dataset, args.dataset_root)
    if args.trials:
        selected = set(open(args.trials).read().split())
        trials = [t for t in trials if t in selected]
    print(args.dataset, len(trials), "trials", flush=True)

    processes = []
    for shard in range(args.shards):
        shard_root = os.path.join(args.work, f"shard_{shard:02d}")
        mirror_root = os.path.join(shard_root, "data", args.dataset)
        for trial in trials[shard::args.shards]:
            target = mirror_trial(args.dataset_root, mirror_root, trial, args.no_caches)
            if args.torso_cache:
                name = f"{args.dataset.lower()}_{trial.replace('/', '_')}.npy"
                shutil.copy2(os.path.join(args.torso_cache, name), os.path.join(target, "obs_torso_shift.npy"))
        command = [sys.executable, os.path.join(REPO, "observations", "extract.py"), "--config", args.config,
                   "--dataset-root", mirror_root, "--results", os.path.join(shard_root, "results")]
        processes.append(subprocess.Popen(command, cwd=REPO, stdout=open(os.path.join(shard_root, "extract.log"), "w"),
                                          stderr=subprocess.STDOUT))
    print("launched", len(processes), "shards", flush=True)
    exit_codes = [p.wait() for p in processes]
    print("exit codes", exit_codes, flush=True)

    combined = os.path.join(args.work, "data")
    os.makedirs(combined, exist_ok=True)
    for shard in range(args.shards):
        results = os.path.join(args.work, f"shard_{shard:02d}", "results")
        for run in os.listdir(results):
            data_dir = os.path.join(results, run, "data")
            for name in os.listdir(data_dir) if os.path.isdir(data_dir) else []:
                target = os.path.join(combined, name)
                if not os.path.lexists(target):
                    os.symlink(os.path.join(data_dir, name), target)
    print("pkls", len(os.listdir(combined)), flush=True)
    # a failed shard leaves its trials out of <work>/data; the exit status makes a driver stop there
    return 1 if any(exit_codes) else 0


if __name__ == "__main__":
    sys.exit(main())
