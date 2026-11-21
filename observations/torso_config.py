"""Write the extraction config that computes only the torso_shift observation and its two filter rows.

The torso rows are added to pickles of the other eight observations (estimate/merge_torso_rows.py). The
config keeps the dataset config's filter parameters for each head.

usage: torso_config.py <dataset config> <out config>
"""
import json
import sys

config = json.load(open(sys.argv[1]))
heads = {m["name"].split("__")[1]: m["params"] for m in config["methods"] if isinstance(m, dict)}
config["methods"] = ["torso_shift"] + [{"name": f"torso_shift__{head}", "params": params} for head, params in heads.items()]
config["name"] += "_torso_rows"
json.dump(config, open(sys.argv[2], "w"), indent=2)
