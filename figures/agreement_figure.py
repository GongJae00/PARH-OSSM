"""Figure 3: how far each method follows the differences in respiratory rate between trials (Hilbert reference).

a, b: trial-mean estimate against trial-mean reference, PARH-OSSM and OSSM-KF, with each method's least-squares
line, the identity line, and the cohort rate that the constant predictor outputs; COHFACE (all 160 trials) and the
MAHNOB-HCI test participants (310 trials). c: mean signed error (bias) of each method in the terciles of the
reference rate on the MAHNOB-HCI test participants, computed as Supplementary Tables 1-3 compute it: the tercile
of a trial comes from its PARH-OSSM trial-mean reference, so every method is summarized on the same trials.
The values come from rate_agreement.py's trial-means CSVs.

usage: agreement_figure.py <COHFACE trial means csv> <MAHNOB-HCI trial means csv> <MAHNOB-HCI subject split csv> <out.pdf>
"""
import sys

import matplotlib
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from figure_style import GUIDE, INK, MM, NEUTRAL, OSSM, PARH

matplotlib.use("pdf")
REFERENCE = "hilbert"
TERCILES = ["low", "middle", "high"]
SCATTER = [("ossm_kf_track", "OSSM-KF", OSSM), ("parh_track", "PARH-OSSM", PARH)]
# panel c: method, name, colour, marker, dashes
BIAS = [("direct_p1d_welch", "Direct P1D quadratic", "#5a5a5a", "s", "-"),
        ("median_fusion", "Median fusion", "#9a9a9a", "^", "-"),
        ("ossm_kf_track", "OSSM-KF", OSSM, "o", "-"),
        ("parh_track", "PARH-OSSM", PARH, "o", "-"),
        ("constant", "Constant predictor", NEUTRAL, None, (0, (1, 1.2)))]
WHITE_EDGE = [patheffects.withStroke(linewidth=1.8, foreground="white")]

cohface_csv, mahnob_csv, split_csv, out_path = sys.argv[1:]
split = pd.read_csv(split_csv)
test = set(split[split.split == "test"].trial)
cohface = pd.read_csv(cohface_csv)
mahnob = pd.read_csv(mahnob_csv)
cohface = cohface[cohface.reference == REFERENCE]
mahnob = mahnob[(mahnob.reference == REFERENCE) & mahnob.trial.isin(test)].copy()

WIDTH, HEIGHT = 175.9, 64.0
fig = plt.figure(figsize=(WIDTH * MM, HEIGHT * MM))


def axes_mm(x0, y0, x1, y1):
    return fig.add_axes((x0 / WIDTH, 1 - y1 / HEIGHT, (x1 - x0) / WIDTH, (y1 - y0) / HEIGHT))


def letter(x_mm, y_mm, text, title):
    fig.text(x_mm / WIDTH, 1 - y_mm / HEIGHT, text, fontsize=8, fontweight="bold", va="baseline")
    fig.text((x_mm + 4.0) / WIDTH, 1 - y_mm / HEIGHT, title, fontsize=6.5, va="baseline")


TOP, BOTTOM = 8.0, 54.0
# one square range for both panels, so that the slopes can be compared by eye
shown_all = pd.concat([t[t.method.isin([m for m, _, _ in SCATTER])] for t in (cohface, mahnob)])
low = np.floor(shown_all[["estimate_bpm", "reference_bpm"]].min().min()) - 0.5
high = np.ceil(shown_all[["estimate_bpm", "reference_bpm"]].max().max()) + 0.5
span = high - low
for panel, (trials, title, x0) in enumerate([(cohface, "COHFACE", 13.0), (mahnob, "MAHNOB-HCI test", 72.0)]):
    ax = axes_mm(x0, TOP, x0 + (BOTTOM - TOP), BOTTOM)
    cohort = trials[trials.method == "constant"].estimate_bpm.iloc[0]
    ax.plot([low, high], [low, high], color=GUIDE, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(cohort, color=NEUTRAL, lw=0.7, ls=(0, (1, 1.2)), zorder=1)
    keys = []
    for method, name, colour in SCATTER:
        g = trials[trials.method == method]
        fit = stats.linregress(g.reference_bpm, g.estimate_bpm)
        ax.scatter(g.reference_bpm, g.estimate_bpm, s=3.2, color=colour, alpha=0.45, lw=0, zorder=2)
        ax.plot([low, high], [fit.intercept + fit.slope * low, fit.intercept + fit.slope * high], color=colour, lw=1.0,
                zorder=4, path_effects=WHITE_EDGE)
        keys.append((name, colour, fit.slope))
    ax.set(xlim=(low, high), ylim=(low, high), aspect="equal")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.tick_params(length=1.5, pad=1, labelsize=5)
    ax.set_xlabel("Reference rate (bpm)", fontsize=5.5, labelpad=1)
    ax.set_ylabel("Estimated rate (bpm)", fontsize=5.5, labelpad=1)
    # key in the empty lower right (high reference, low estimate): a stroke of each fit line, its name, its slope
    for k, (name, colour, slope) in enumerate(keys):
        y = low + span * (0.06 + 0.08 * k)
        x = low + span * 0.44
        ax.plot([x, x + span * 0.06], [y, y], color=colour, lw=1.0)
        ax.text(x + span * 0.085, y, f"{name}, slope {slope:.2f}", fontsize=5.2, va="center", color=INK)
    ax.text(low + span * 0.02, cohort + span * 0.012, "cohort rate", fontsize=4.8, color="#6f6f6f", ha="left",
            va="bottom", path_effects=WHITE_EDGE)
    letter(x0 - 11.0, 4.6, "ab"[panel], title)

# c: bias by tercile of the reference rate, as in Supplementary Tables 1-3
parh = mahnob[mahnob.method == "parh_track"].set_index("trial").reference_bpm
edges = parh.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
mahnob["tercile"] = mahnob.trial.map(pd.cut(parh, edges, labels=TERCILES, include_lowest=True))
bias = mahnob.groupby(["method", "tercile"], observed=True).difference_bpm.mean()
x0 = 128.0
ax = axes_mm(x0, TOP, x0 + 24.0, BOTTOM)
ax.axhline(0, color=GUIDE, lw=0.5, zorder=1)
ends = []
for method, name, colour, marker, dashes in BIAS:
    values = [bias.loc[(method, t)] for t in TERCILES]
    ax.plot(range(3), values, color=colour, lw=0.9, ls=dashes, marker=marker, ms=2.4, mew=0, zorder=3)
    ends.append((values[-1], name, colour))
    print(f"bias {name}: " + ", ".join(f"{t} {v:+.2f}" for t, v in zip(TERCILES, values)))
ax.set_xlim(-0.3, 2.3)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{t.capitalize()}\n{edges[k]:.1f}–{edges[k + 1]:.1f}" for k, t in enumerate(TERCILES)])
ax.tick_params(length=1.5, pad=1, labelsize=5)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
ax.set_ylabel("Bias, estimate minus reference (bpm)", fontsize=5.5, labelpad=1)
ax.set_xlabel("Tercile of the reference rate (bpm)", fontsize=5.5, labelpad=1)
lo, hi = ax.get_ylim()
ax.set_ylim(lo, hi)
# direct labels at the right ends, spread to at least 2.4 mm
height_mm = BOTTOM - TOP
order = np.argsort([e[0] for e in ends])
ys = np.array([e[0] for e in ends], float)
gap = 2.4 * (hi - lo) / height_mm
y = ys[order].copy()
for _ in range(60):
    for k in range(1, len(y)):
        if y[k] - y[k - 1] < gap:
            shift = (gap - (y[k] - y[k - 1])) / 2
            y[k - 1] -= shift
            y[k] += shift
placed = np.empty_like(y)
placed[order] = y
for (value, name, colour), label_y in zip(ends, placed):
    ax.annotate(name, xy=(2, value), xytext=(2.45, label_y), fontsize=5.2, color=INK, va="center",
                annotation_clip=False, arrowprops=dict(arrowstyle="-", color=colour, lw=0.5, shrinkA=0, shrinkB=1.5))
letter(x0 - 11.0, 4.6, "c", "MAHNOB-HCI test")
for side in ("top", "right"):
    for a in fig.axes:
        a.spines[side].set_visible(False)

fig.savefig(out_path)
print(f"edges {edges.round(2)}")
