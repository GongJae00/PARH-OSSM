"""LaTeX bodies of the Supplementary tables built from the stored per-trial records.

Rate terciles, one table per reference definition: per tercile of the trial-mean reference
rate, the median trial MAE and the mean bias (estimate minus reference) of each method, from
rate_agreement.py's trial-means CSVs. Motion tertiles: the median trial MAE per tertile of head
motion under each reference, from motion_tertiles.py's per-trial CSVs. Perturbations: the median trial
MAE per condition under each reference, from motion_perturbation_summary.py's per-trial CSV. Redundancy: the
waveform similarity and rate agreement of the observation pairs, from channel_redundancy.py, and the change in
error when part of the bank is left out, scored by reliability_controls_summary.compare. Reliability controls
and the confirmatory tests, scored the same way. The rows of the main Tables 2 and 3 and of their MAHNOB-HCI
counterparts, split from the outputs of main_tables.sh. Writing the rows from the records keeps the typeset numbers
equal to the analysis outputs.

usage: supplementary_tables.py <records dir> <out dir>
<records dir> holds the CSVs the evaluation scripts wrote, under the names used below, and the MAHNOB-HCI subject split
(evaluation/mahnob_subject_split.csv).
"""
import os
import sys

import pandas as pd
from scipy.stats import false_discovery_control

from reliability_controls_summary import compare

REFERENCES = [("hilbert", "Hilbert"), ("welch", "Welch"), ("count", "Breath count")]
METHODS = [("direct_p1d_welch", "Direct P1D quadratic"), ("median_fusion", "Median fusion"),
           ("ossm_kf_track", "OSSM-KF"), ("parh_track", "PARH-OSSM"), ("constant", "Constant predictor")]
# the frozen method: COHFACE on the first-frame observations, MAHNOB-HCI under the onset placement
DATASETS = [("COHFACE", "cohface_frozen"), ("MAHNOB-HCI", "mahnob_frozen_onset")]
# motion_tertiles.py scores the methods of the window-rate csv, which holds no median fusion
MOTION_METHODS = [m for m in METHODS if m[0] != "median_fusion"]
TERCILES = ["low", "middle", "high"]
records_dir, out_dir = sys.argv[1:3]


def number(x, signed=False, digits=2):
    return (f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}").replace("-", "$-$").replace("+", "$+$")


for reference, _ in REFERENCES:
    lines = []
    for dataset, prefix in DATASETS:
        trials = pd.read_csv(os.path.join(records_dir, "rate_agreement", f"{prefix}_trial_means.csv"))
        trials = trials[trials.reference == reference].copy()
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
    motion = trials.drop_duplicates("trial").groupby("tertile").motion_range.agg(["min", "max"]).reindex(TERCILES)
    ranges = ", ".join(f"{motion.loc[t, 'min']:.1f}--{motion.loc[t, 'max']:.1f}" for t in TERCILES)
    lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{dataset}}}, {trials.trial.nunique()} trials}} \\\\")
    lines.append(f"\\multicolumn{{10}}{{l}}{{\\quad Head motion of the low, middle, and high tertiles: {ranges}\\,\\% of the face height}} \\\\")
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
# Redundancy: per pair of observations, the waveform similarity and the median |rate difference| over windows
# from channel_redundancy.py, summarised per relation as the median similarity over pair-trials and the share of
# pair-trials whose rates agree within 1 bpm; pairs with a flat observation (MAHNOB-HCI trials without P1D) are left out
FAMILY = {"of_farneback": "OF", "of_disp_bridge": "OF", "DoF": "DoF", "dof_disp_bridge": "DoF", "profile1D linear": "P1D",
          "profile1D quadratic": "P1D", "profile1D cubic": "P1D", "profile1d_consensus": "P1D", "torso_shift": "Torso"}
RELATIONS = [("OF-OF", "Within OF: OF and OF bridge"), ("DoF-DoF", "Within DoF: DoF and DoF bridge"),
             ("P1D quadratic-cubic", "Within P1D: quadratic and cubic"), ("P1D-P1D", "Within P1D: other pairs"),
             ("OF-P1D", "OF and P1D"), ("DoF-OF", "OF and DoF"), ("DoF-P1D", "DoF and P1D"),
             ("OF-Torso", "Torso and OF"), ("DoF-Torso", "Torso and DoF"), ("P1D-Torso", "Torso and P1D")]
REDUNDANCY_DATASETS = [("COHFACE", "cohface"), ("MAHNOB-HCI", "mahnob")]
summaries = []
for _, prefix in REDUNDANCY_DATASETS:
    pairs = pd.read_csv(os.path.join(records_dir, f"{prefix}_channel_redundancy.csv")).dropna()
    quadratic_cubic = (pairs.channel_a == "profile1D quadratic") & (pairs.channel_b == "profile1D cubic")
    pairs["relation"] = ["-".join(sorted((FAMILY[a], FAMILY[b]))) for a, b in zip(pairs.channel_a, pairs.channel_b)]
    pairs.loc[quadratic_cubic, "relation"] = "P1D quadratic-cubic"
    summaries.append(pairs.groupby("relation").agg(similarity=("waveform_similarity", "median"),
                                                   agree=("rate_gap_bpm", lambda g: 100 * (g < 1).mean())))
lines = []
for relation, name in RELATIONS:
    cells = " & ".join(f"{summary.loc[relation, 'similarity']:.2f} & {summary.loc[relation, 'agree']:.0f}" for summary in summaries)
    lines.append(f"{name} & {cells} \\\\")
with open(os.path.join(out_dir, "S_redundancy.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
# Family exclusion: PARH-OSSM rerun with part of the bank left out, against the frozen bank, scored as the
# confirmatory tests score (reliability_controls_summary.compare): per reference, the median over subjects of the
# per-subject median of (left-out MAE - frozen MAE), positive when the left-out observations lowered the error, and
# the number of subjects with a positive value. MAHNOB-HCI rows are the test subjects under the primary placement.
EXCLUSIONS = [("no_torso", "Torso displacement"), ("no_of", "OF family"), ("no_dof", "DoF family"), ("no_p1d", "P1D family"),
              ("no_bridges", "OF and DoF bridges"), ("no_p1d_cubic", "P1D cubic"), ("no_p1d_consensus", "P1D consensus")]
EXCLUSION_DATASETS = [("cohface_frozen_window_rates_three_references.csv", "none", "all",
                       "cohface_loo9_{}_window_rates_three_references.csv"),
                      ("mahnob_frozen_onset_window_rates_three_references.csv", "mahnob_subject_split.csv", "test",
                       "mahnob_loo9_{}_onset_window_rates_three_references.csv")]


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
                    ("mahnob_frozen_onset_window_rates_three_references.csv", "mahnob_subject_split.csv", "test",
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
    tests = compare(os.path.join(records_dir, f"{frozen}_window_rates_three_references.csv"), os.path.join(records_dir, "mahnob_subject_split.csv"), "test",
                    [f"submitted={os.path.join(records_dir, submitted + '_window_rates_three_references.csv')}:parh_track_bpm"])[3]
    tests["q"] = false_discovery_control(tests.p)
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}} \\\\")
    for comparator, name in [("constant", "Constant predictor"), ("submitted", "PARH-OSSM, chest observations only")]:
        for r, reference_name in REFERENCES:
            t = tests[(tests.comparator == comparator) & (tests.reference == r)].iloc[0]
            lines.append(f"\\quad {name} & {reference_name} & {number(t['subject median gain'], True)} & {t['subjects better']} & {t.p:.3f} & {t.q:.3f} \\\\")
    lines.append("\\addlinespace")
with open(os.path.join(out_dir, "S_confirmatory.tex"), "w") as f:
    f.write("\n".join(lines[:-1]) + "\n")

# Rows of the main Tables 2 and 3 (T_*) and of their MAHNOB-HCI counterparts in the Supplementary (S_*), from the
# outputs of main_tables.sh: COHFACE in cohface_main_tables.tex; MAHNOB-HCI in mahnob_main_tables.tex,
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
mahnob = main_table_blocks("mahnob_main_tables.tex")
for table, name in [("2", "rate"), ("3", "stages")]:
    write_rows(f"T_{name}_cohface.tex", cohface[(None, "all", table)]["rows"])
    write_rows(f"T_{name}_mahnob.tex", mahnob[("onset", "test", table)]["rows"])
    lines = []
    for placement, split in [("onset", "development"), ("onset", "all"), ("tail", "test"), ("tail", "development"), ("tail", "all")]:
        block = mahnob[(placement, split, table)]
        position = "starts at the stimulus onset" if placement == "onset" else "ends with the recording"
        who = {"test": "test participants", "development": "development participants", "all": "all participants"}[split]
        lines.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{Video {position}; {block['subjects']} {who}, {block['trials']} trials}}}} \\\\")
        lines += block["rows"] + ["\\addlinespace"]
    write_rows(f"S_{name}_mahnob.tex", lines[:-1])
print(f"wrote {len(REFERENCES) + 12} table bodies to {out_dir}")
