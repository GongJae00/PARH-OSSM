"""LaTeX bodies of the Supplementary tables built from the stored per-trial records.

Rate terciles, one table per reference definition: per tercile of the trial-mean reference
rate, the median trial MAE and the mean bias (estimate minus reference) of each method, from
rate_agreement.py's trial-means CSVs. Motion tertiles: the median trial MAE per tertile of head
motion under each reference, from motion_tertiles.py's per-trial CSVs. Perturbations: the median trial
MAE per condition under each reference, from motion_perturbation_summary.py's per-trial CSV. Redundancy: the
breathing-band coherence and the correlation above the breathing band of the observation pairs, from
observation_error_correlation.py, and the change in error when a measurement route is left out of the bank, scored by
reliability_controls_summary.compare. Reliability controls and the confirmatory tests, scored the same way. The rows
of the main Tables 2 and 3 and of their MAHNOB-HCI counterparts, split from the outputs of main_tables.sh. Writing the
rows from the records keeps the typeset numbers equal to the analysis outputs.

MAHNOB-HCI rows are those of the evaluated split, the test participants for the reported tables; scoring the
development participants instead runs the whole chain without a test score. The route exclusions are the
measurement routes left out of the bank, one row each.

usage: supplementary_tables.py <records dir> <out dir> <MAHNOB-HCI split: test|development> <route exclusions, comma-separated>
<records dir> holds the CSVs the evaluation scripts wrote, under the names used below, and the MAHNOB-HCI subject split
(evaluation/mahnob_subject_split.csv).
"""
import os
import sys

import pandas as pd
from scipy.stats import false_discovery_control

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "estimate"))
from bank import ADDED_BACK, ROUTES as ROUTE_ROWS
from reliability_controls_summary import compare

records_dir, out_dir, MAHNOB_EVALUATED, ROUTES = sys.argv[1:5]
ROUTE_NAMES = {name: row for name, (_, row) in {**ROUTE_ROWS, **ADDED_BACK}.items()}
REFERENCES = [("hilbert", "Hilbert"), ("welch", "Welch"), ("count", "Breath count")]
METHODS = [("direct_p1d_welch", "Direct P1D quadratic"), ("median_fusion", "Median fusion"),
           ("ossm_kf_track", "OSSM-KF"), ("parh_track", "PARH-OSSM"), ("constant", "Constant predictor")]
# the frozen method: COHFACE on the first-frame observations, MAHNOB-HCI under the onset placement
DATASETS = [("COHFACE", "cohface_frozen"), ("MAHNOB-HCI", "mahnob_frozen_onset")]
split = pd.read_csv(os.path.join(records_dir, "mahnob_subject_split.csv"))
MAHNOB_TRIALS = set(split[split.split == MAHNOB_EVALUATED].trial)
# motion_tertiles.py scores the methods of the window-rate csv, which holds no median fusion
MOTION_METHODS = [m for m in METHODS if m[0] != "median_fusion"]
TERCILES = ["low", "middle", "high"]


def number(x, signed=False, digits=2):
    return (f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}").replace("-", "$-$").replace("+", "$+$")


for reference, _ in REFERENCES:
    lines = []
    for dataset, prefix in DATASETS:
        trials = pd.read_csv(os.path.join(records_dir, "rate_agreement", f"{prefix}_trial_means.csv"))
        trials = trials[trials.reference == reference].copy()
        if dataset == "MAHNOB-HCI":
            trials = trials[trials.trial.isin(MAHNOB_TRIALS)]
        edges = trials[trials.method == "parh_track"].reference_bpm.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
        trials["tercile"] = pd.cut(trials.reference_bpm, edges, labels=TERCILES, include_lowest=True)
        ranges = " & ".join(f"\\multicolumn{{2}}{{c}}{{{edges[k]:.1f}--{edges[k + 1]:.1f}}}" for k in range(3))
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{dataset}}}, {trials.trial.nunique()} trials}} \\\\")
        lines.append(f"\\quad Reference rate (bpm) & {ranges} \\\\")
        summary = trials.groupby(["method", "tercile"], observed=True).agg(MAE=("MAE", "median"), bias=("difference_bpm", "mean"))
        for method, name in METHODS:
            cells = " & ".join(f"{number(summary.loc[(method, t), 'MAE'])} & {number(summary.loc[(method, t), 'bias'], True)}"
                               for t in TERCILES)
            lines.append(f"\\quad {name} & {cells} \\\\")
        lines.append("\\addlinespace")
    with open(os.path.join(out_dir, f"S_rate_terciles_{reference}.tex"), "w") as f:
        f.write("\n".join(lines[:-1]) + "\n")

lines = []
for dataset, prefix in DATASETS:
    trials = pd.read_csv(os.path.join(records_dir, "rate_agreement", f"{prefix}_head_motion_trials.csv"))
    if dataset == "MAHNOB-HCI":
        trials = trials[trials.trial.isin(MAHNOB_TRIALS)]
    motion = trials.drop_duplicates("trial").groupby("tertile").motion_range.agg(["min", "max"]).reindex(TERCILES)
    ranges = ", ".join(f"{motion.loc[t, 'min']:.1f}--{motion.loc[t, 'max']:.1f}" for t in TERCILES)
    lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{dataset}}}, {trials.trial.nunique()} trials}} \\\\")
    lines.append(f"\\multicolumn{{10}}{{l}}{{\\quad Head motion of the low, middle, and high terciles: {ranges}\\,\\% of the face height}} \\\\")
    median = trials.groupby(["method", "reference", "tertile"]).MAE.median()
    for method, name in MOTION_METHODS:
        cells = " & ".join(number(median.loc[(method, r, t)]) for r, _ in REFERENCES for t in TERCILES)
        lines.append(f"\\quad {name} & {cells} \\\\")
    lines.append("\\addlinespace")
with open(os.path.join(out_dir, "S_motion_tertiles.tex"), "w") as f:
    f.write("\n".join(lines[:-1]) + "\n")

PERTURBATIONS = [("none", "None"),
                 ("translation:0.5", "Translation 0.5\\,px"), ("translation:1", "Translation 1\\,px"), ("translation:2", "Translation 2\\,px"),
                 ("rotation:0.25", "Rotation 0.25$^\\circ$"), ("rotation:0.5", "Rotation 0.5$^\\circ$"), ("rotation:1", "Rotation 1$^\\circ$"),
                 ("brightness:0.01", "Brightness 1\\,\\%"), ("brightness:0.02", "Brightness 2\\,\\%"), ("brightness:0.05", "Brightness 5\\,\\%")]
PERTURBED_METHODS = ["direct P1D", "OSSM-KF", "PARH-OSSM"]
perturbation = pd.read_csv(os.path.join(records_dir, "cohface_perturbation_trial_mae.csv")).groupby(["condition", "reference", "method"]).MAE.median()
lines = []
for condition, name in PERTURBATIONS:
    cells = " & ".join(number(perturbation.loc[(condition, r, m)]) for r, _ in REFERENCES for m in PERTURBED_METHODS)
    lines.append(f"{name} & {cells} \\\\")
with open(os.path.join(out_dir, "S_perturbation.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
# Redundancy: per pair of observations, from observation_error_correlation.py on every trial (it reads no
# reference): the median over trials of the breathing-band magnitude-squared coherence, which is near 1 when one
# observation is a linear filter of the other whatever the phase, and of the absolute correlation above the breathing
# band at the lag the filter's alignment picks, which estimates how far the errors of the pair are shared. The first
# rows are the pairs the bank treats as one measurement, then the two statistics of the frame difference, then the
# pairs of the bank's different measurements pooled by route; trials with a flat observation are left out.
REDUNDANCY_ROWS = [
    ("One measurement", [
        ("OF and OF bridge", [("of_farneback", "of_disp_bridge")]),
        ("P1D quadratic and cubic", [("profile1D quadratic", "profile1D cubic")]),
        ("P1D quadratic and consensus", [("profile1D quadratic", "profile1d_consensus")]),
        ("P1D cubic and consensus", [("profile1D cubic", "profile1d_consensus")]),
        ("P1D linear and quadratic", [("profile1D linear", "profile1D quadratic")]),
        ("P1D linear and cubic", [("profile1D linear", "profile1D cubic")]),
        ("P1D linear and consensus", [("profile1D linear", "profile1d_consensus")])]),
    ("Two statistics of one frame difference", [
        ("DoF and DoF bridge", [("DoF", "dof_disp_bridge")])]),
    ("Different measurements", [
        ("OF and DoF statistics", [("of_farneback", "DoF"), ("of_farneback", "dof_disp_bridge")]),
        ("OF and P1D quadratic", [("of_farneback", "profile1D quadratic")]),
        ("DoF statistics and P1D quadratic", [("DoF", "profile1D quadratic"), ("dof_disp_bridge", "profile1D quadratic")]),
        ("Torso displacement and OF", [("of_farneback", "torso_shift")]),
        ("Torso displacement and DoF statistics", [("DoF", "torso_shift"), ("dof_disp_bridge", "torso_shift")]),
        ("Torso displacement and P1D quadratic", [("profile1D quadratic", "torso_shift")])]),
]
REDUNDANCY_FILES = ["cohface_error_correlation.csv", "mahnob_error_correlation.csv"]
pair_tables = []
for file_name in REDUNDANCY_FILES:
    pairs = pd.read_csv(os.path.join(records_dir, file_name)).dropna()
    pairs["pair"] = [frozenset((a, b)) for a, b in zip(pairs.a, pairs.b)]
    pairs["shared error"] = pairs["above, filter lag"].abs()
    pair_tables.append(pairs)
lines = []
for group, rows in REDUNDANCY_ROWS:
    lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{group}}}}} \\\\")
    for name, members in rows:
        cells = []
        for pairs in pair_tables:
            pooled = pairs[pairs.pair.isin([frozenset(m) for m in members])]
            cells.append(f"{pooled['breathing coherence'].median():.2f} & {pooled['shared error'].median():.2f}")
        lines.append(f"\\quad {name} & {' & '.join(cells)} \\\\")
with open(os.path.join(out_dir, "S_redundancy.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
# Route exclusion: PARH-OSSM rerun with part of the bank left out, against the frozen bank, scored as the
# confirmatory tests score (reliability_controls_summary.compare): per reference, the median over subjects of the
# per-subject median of (left-out MAE - frozen MAE), positive when the left-out observations lowered the error, and
# the number of subjects with a positive value. MAHNOB-HCI rows are the test subjects under the primary placement.
EXCLUSIONS = [(name, ROUTE_NAMES[name]) for name in ROUTES.split(",")]
EXCLUSION_DATASETS = [("cohface_frozen_window_rates_three_references.csv", "none", "all",
                       "cohface_{}_window_rates_three_references.csv"),
                      ("mahnob_frozen_onset_window_rates_three_references.csv", "mahnob_subject_split.csv", MAHNOB_EVALUATED,
                       "mahnob_{}_onset_window_rates_three_references.csv")]


def gains_against(variants, datasets):
    """Per dataset, the frozen method against each variant: the subject-level gains of compare()."""
    gains = []
    for frozen, split_csv, evaluated, variant in datasets:
        comparators = [f"{v}={os.path.join(records_dir, variant.format(v))}:parh_track_bpm" for v, _ in variants]
        split_path = split_csv if split_csv == "none" else os.path.join(records_dir, split_csv)
        gains.append(compare(os.path.join(records_dir, frozen), split_path, evaluated, comparators)[3].set_index(["comparator", "reference"]))
    return gains


def write_gain_rows(path, variants, gains):
    lines = []
    for variant, name in variants:
        cells = " & ".join(f"{number(g.loc[(variant, r), 'subject median gain'], True, 3)} ({g.loc[(variant, r), 'subjects better'].split('/')[0]})"
                           for g in gains for r, _ in REFERENCES)
        lines.append(f"{name} & {cells} \\\\")
    with open(os.path.join(out_dir, path), "w") as f:
        f.write("\n".join(lines) + "\n")


write_gain_rows("S_exclusion.tex", EXCLUSIONS, gains_against(EXCLUSIONS, EXCLUSION_DATASETS))

# Reliability controls: the trust of each observation in the filter replaced by its mean over the observations
# at the same frame (equal_filter_trust.py), and the readout's reliability priors made equal across families or shuffled
# in time within each trial (reliability_controls.py)
CONTROLS = [("equal_filter", "Filter trust equal across observations"), ("equal_readout", "Readout priors equal across families"),
            ("shuffled_readout", "Readout priors shuffled in time")]
CONTROL_DATASETS = [("cohface_frozen_window_rates_three_references.csv", "none", "all",
                     "cohface_{}_window_rates_three_references.csv"),
                    ("mahnob_frozen_onset_window_rates_three_references.csv", "mahnob_subject_split.csv", MAHNOB_EVALUATED,
                     "mahnob_{}_onset_window_rates_three_references.csv")]
write_gain_rows("S_reliability.tex", CONTROLS, gains_against(CONTROLS, CONTROL_DATASETS))

# Confirmatory tests, declared before scoring: on the MAHNOB-HCI test subjects, the frozen
# method against the submitted method (run w2, chest observations only) and against the constant predictor, under
# each reference; the six p values of a placement are one Benjamini-Hochberg family. The gains are those of
# confirmatory_test.py, computed on the same trials by compare().
PLACEMENTS = [("onset", "Video starts at the stimulus onset", "mahnob_frozen_onset", "mahnob_w2_onset"),
              ("tail", "Video ends with the recording", "mahnob_frozen", "mahnob_w2")]
lines = []
for _, label, frozen, submitted in PLACEMENTS:
    tests = compare(os.path.join(records_dir, f"{frozen}_window_rates_three_references.csv"), os.path.join(records_dir, "mahnob_subject_split.csv"), MAHNOB_EVALUATED,
                    [f"submitted={os.path.join(records_dir, submitted + '_window_rates_three_references.csv')}:parh_track_bpm"])[3]
    tests["q"] = false_discovery_control(tests.p)
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}} \\\\")
    for comparator, name in [("constant", "Constant predictor"), ("submitted", "PARH-OSSM as submitted")]:
        for r, reference_name in REFERENCES:
            t = tests[(tests.comparator == comparator) & (tests.reference == r)].iloc[0]
            lines.append(f"\\quad {name} & {reference_name} & {number(t['subject median gain'], True)} & {t['subjects better']} & {t.p:.3f} & {t.q:.3f} \\\\")
    lines.append("\\addlinespace")
with open(os.path.join(out_dir, "S_confirmatory.tex"), "w") as f:
    f.write("\n".join(lines[:-1]) + "\n")

# Rows of the main Tables 2 and 3 (T_*) and of their MAHNOB-HCI counterparts in the Supplementary (S_*), from the
# outputs of main_tables.sh: COHFACE in cohface_main_tables.tex; MAHNOB-HCI in mahnob_<split>_main_tables.tex,
# whose blocks follow a "% placement" line, a "% Table" line, and the dataset line that names the evaluated split.
def main_table_blocks(path):
    blocks, placement, table = {}, None, None
    for line in open(os.path.join(records_dir, path)).read().splitlines():
        if line.startswith("% placement"):
            placement = line.split()[-1]
        elif line.startswith("% Table"):
            table = line.split()[-1]
        elif line.startswith("% "):
            split = line.split("(split ")[1].split(")")[0]
            trials, subjects = line.split(": ")[1].split(" trials, ")[0], line.split(" trials, ")[1].split(" subjects")[0]
            block = blocks.setdefault((placement, split, table), {"trials": trials, "subjects": subjects, "rows": []})
        else:
            block["rows"].append(line.replace("+ ", "\\;$+$ ") if line.startswith("+ ") else line)
    return blocks


def write_rows(path, lines):
    with open(os.path.join(out_dir, path), "w") as f:
        f.write("\n".join(lines) + "\n")


cohface = main_table_blocks("cohface_main_tables.tex")
mahnob = main_table_blocks(f"mahnob_{MAHNOB_EVALUATED}_main_tables.tex")
for table, name in [("2", "rate"), ("3", "stages")]:
    write_rows(f"T_{name}_cohface.tex", cohface[(None, "all", table)]["rows"])
    write_rows(f"T_{name}_mahnob.tex", mahnob[("onset", MAHNOB_EVALUATED, table)]["rows"])
    lines = []
    blocks = [(p, s) for p in ["onset", "tail"] for s in ["test", "development", "all"]]
    for placement, split in [b for b in blocks if b != ("onset", MAHNOB_EVALUATED) and (*b, table) in mahnob]:
        block = mahnob[(placement, split, table)]
        position = "starts at the stimulus onset" if placement == "onset" else "ends with the recording"
        who = {"test": "test participants", "development": "development participants", "all": "all participants"}[split]
        lines.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{Video {position}; {block['subjects']} {who}, {block['trials']} trials}}}} \\\\")
        lines += block["rows"] + ["\\addlinespace"]
    write_rows(f"S_{name}_mahnob.tex", lines[:-1])
print(f"wrote {len(REFERENCES) + 12} table bodies to {out_dir}")
