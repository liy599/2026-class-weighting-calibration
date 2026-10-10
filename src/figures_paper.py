"""
Publication figures.

Four figures, each replacing a table the paper does not have room for. They
are built from the cached probability maps and the experiment CSVs, so
nothing here is hand-entered.

Design notes, in case these are edited later:

  * Every figure is drawn AT THE SIZE IT IS PRINTED. A figure drawn six inches
    wide and shrunk to a 3.5 inch column takes its 8 pt text down to 4.5 pt,
    which is what made the first version hard to read. Here the drawing width
    equals the printed width (COL_W for one column, FULL_W for two), so a font
    size written in this file is the font size on the page. The paper must
    include each figure at the width noted in its docstring.
  * Three categorical hues, taken in fixed slot order and validated for
    colour-vision deficiency separation (worst adjacent pair dE 23.1 protan,
    24.0 normal). Identity never rests on colour alone: every loss also has
    its own marker shape and is named in the legend.
  * Text uses ink tokens, never a series colour, so a label is never mistaken
    for a data mark.
  * Vector PDF for the paper, PNG for the notebook. Pure white background.

Run:
    python figures_paper.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config
import metrics as M
import models as MD


# ----------------------------------------------------------------- TOKENS

INK       = "#0b0b0b"     # primary
INK2      = "#52514e"     # secondary
MUTED     = "#6f6d68"     # axis / tick labels (darkened a little so that small
                          # print sizes stay legible)
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
SURFACE   = "#ffffff"     # pure white, to match the page of the paper

LOSS = {                          # colour, marker, label  (fixed slot order)
    "BCE":  ("#1baf7a", "o", "BCE (unweighted)"),
    "DICE": ("#2a78d6", "s", "DICE"),
    "wBCE": ("#eb6834", "D", "wBCE (weighted ×166)"),
}
ORDER = ["BCE", "DICE", "wBCE"]

CONFIGS = ["trainable_guided", "frozen_guided",
           "trainable_unguided", "frozen_unguided"]

POS_WEIGHT = 166.0
N_BINS = 15
BAND_RADIUS = 10

# Printed widths in inches (IEEEtran conference: 3.5 in column, 7.16 in page).
COL_W = 3.5
FULL_W = 0.86 * 7.16          # matches \includegraphics[width=0.86\textwidth]


def _setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42,          # editable text in the PDF
        "ps.fonttype": 42,
    })


def _save(fig, stem, quiet=False):
    """Write both formats and hand the figure back.

    Deliberately does not close the figure: the notebook calls these
    functions directly so that running it regenerates the paper figures
    rather than displaying stale PNGs, and an inline backend needs the
    figure left open to render it. `main()` closes them instead.
    """
    for ext in ("pdf", "png"):
        p = os.path.join(config.FIGURES, f"{stem}.{ext}")
        # A PDF normally records the moment it was written, so regenerating an
        # unchanged figure would still change the file and show up in version
        # control. Dropping the date makes the output reproducible byte for byte.
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03,
                    metadata=meta)
    if not quiet:
        print(f"  wrote {stem}.pdf / .png")
    return fig


def _band(t):
    return M.band_mask(t, radius=BAND_RADIUS)


# ------------------------------------------------- FIG 1  the separation

def fig_separation(per_model, stem="paper_fig1_separation", quiet=False):
    """Signed calibration error, every model, grouped by configuration.

    Include at width = 0.86 \\textwidth (a two column figure).

    The job is 'are the three losses separable', so the form is a dot plot
    with one row per configuration and a connecting rule: the reader checks
    whether the three bands ever overlap, which is a position judgement, not
    a colour one.
    """
    d = (per_model[per_model.split == "seen"]
         .groupby(["config", "loss"]).sce.mean().unstack())

    fig, ax = plt.subplots(figsize=(FULL_W, 3.15))
    ys = np.arange(len(CONFIGS))[::-1]

    for y, cfg in zip(ys, CONFIGS):
        vals = [d.loc[cfg, l] for l in ORDER]
        ax.plot([min(vals), max(vals)], [y, y], "-", color=BASELINE,
                lw=1.1, zorder=1, solid_capstyle="round")
        for l in ORDER:
            colour, marker, _ = LOSS[l]
            ax.plot(d.loc[cfg, l], y, marker, color=colour, ms=9,
                    markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=3)

    # The zero rule is named in the legend rather than annotated in place:
    # the space above it carries the direct labels and the space below it
    # carries the axis title, so an in-place note collides with one or the
    # other at any figure size.
    ax.axvline(0, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)

    ax.set_yticks(ys)
    ax.set_yticklabels([c.replace("_", " + ") for c in CONFIGS],
                       fontsize=10, color=INK2)
    ax.set_xlabel("Signed calibration error near the line "
                  "(positive = overconfident)", fontsize=10, labelpad=7)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xlim(-0.035, 0.235)
    ax.set_ylim(-0.55, len(CONFIGS) - 0.1)
    ax.grid(axis="x", alpha=0.75)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # Direct labels on the top row only; the legend covers the rest.
    # BCE and DICE sit only 0.017 apart, so centred labels touch at this size.
    # Push them apart: BCE's to the left of its marker, DICE's to the right.
    nudge = {"BCE": (-3, "right"), "DICE": (3, "left"), "wBCE": (0, "center")}
    for l in ORDER:
        dx, ha = nudge[l]
        ax.annotate(l, (d.loc[CONFIGS[0], l], ys[0]),
                    textcoords="offset points", xytext=(dx, 12),
                    ha=ha, fontsize=9.5, color=INK2)

    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=8,
                      markeredgecolor=SURFACE, markeredgewidth=1.2, label=lab)
               for c, m, lab in (LOSS[l] for l in ORDER)]
    handles.append(Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 3)),
                          label="perfectly calibrated"))
    # Two columns: four entries in one row would run past the figure width.
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.45, -0.24), ncol=2, fontsize=9.5,
              handletextpad=0.5, columnspacing=2.0, labelspacing=0.45)

    return _save(fig, stem, quiet)


# ------------------------------------------- FIG 2  reliability + theory

def fig_reliability(images, stem="paper_fig2_reliability", quiet=False):
    """Reliability curves, with the weighted-loss optimum drawn as theory.

    Include at width = \\columnwidth.

    Weighted BCE with weight w is minimised at q* = w p / (1 - p + w p), so a
    model that reports q should be correct a fraction q / (w + q - q w) of the
    time. That is a parameter free prediction, plotted as the dashed line.

    The lower panel carries the bin populations, which is not decoration: it
    is what shows that DICE hardly uses the middle of the probability range
    at all, so its small signed error reflects near binary output rather than
    genuine calibration.
    """
    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=(COL_W, 4.05), sharex=True,
        gridspec_kw=dict(height_ratios=[3.0, 1.0], hspace=0.14))

    # The diagonal is named in the legend rather than inline: BCE lies almost
    # on top of it, so any inline label would sit on a data mark.
    ax.plot([0, 1], [0, 1], "-", color=BASELINE, lw=1.3, zorder=1,
            label="perfect calibration")

    short = {"BCE": "BCE", "DICE": "DICE", "wBCE": "wBCE"}
    for l in ORDER:
        colour, marker, _ = LOSS[l]
        p, y = MD.pooled_pixels(MD.tags_with(loss=l), images, "seen", _band)
        r = M.reliability_curve(p, y, N_BINS)
        ece = M.expected_calibration_error(p, y, N_BINS)
        ax.plot(r["confidence"], r["accuracy"], "-", marker=marker,
                color=colour, lw=1.9, ms=5, markeredgecolor=SURFACE,
                markeredgewidth=0.8, zorder=3,
                label=f"{short[l]}  (ECE {ece:.3f})")
        axc.plot(r["confidence"], r["count"], "-", marker=marker,
                 color=colour, lw=1.4, ms=4, markeredgecolor=SURFACE,
                 markeredgewidth=0.6)

    q = np.linspace(0.001, 0.999, 300)
    ax.plot(q, q / (POS_WEIGHT + q - q * POS_WEIGHT), ls=(0, (5, 3)),
            color=INK2, lw=1.4, zorder=2, label="wBCE theory")

    ax.set_ylabel("Observed frequency of an edge", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", handletextpad=0.5, labelspacing=0.3,
              fontsize=7.8, borderaxespad=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    axc.set_yscale("log")
    axc.set_ylabel("Pixels per bin", fontsize=8.5)
    axc.set_xlabel("Predicted probability", fontsize=9)
    axc.grid(alpha=0.75)
    axc.set_axisbelow(True)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    return _save(fig, stem, quiet)


# ----------------------------------------- FIG 3  effect of the weight on the threshold

def fig_threshold(threshold, stem="paper_fig3_threshold", quiet=False):
    """Where each loss is usable on the decision threshold axis.

    Include at width = \\columnwidth.

    The bar is the operating window: the span over which FOM stays within
    0.02 of that loss's own best. The marker is the optimum, the dashed line is
    the default cut, and the percentage on the right is the width of the bar
    as a share of all thresholds. The reader's question is 'does the default
    cut fall inside the bar', which is a position judgement. A legend names
    the bar, the marker and the dashed line.
    """
    d = (threshold.groupby("loss")
         .agg(lo=("window_lo", "mean"), hi=("window_hi", "mean"),
              opt=("best_threshold", "mean"), frac=("window_frac", "mean"),
              fom=("fom_best", "mean"))
         .reindex(ORDER))

    fig, ax = plt.subplots(figsize=(COL_W - 0.45, 2.35))
    ys = np.arange(len(ORDER))[::-1]

    for y, l in zip(ys, ORDER):
        colour, marker, _ = LOSS[l]
        r = d.loc[l]
        ax.barh(y, r.hi - r.lo, left=r.lo, height=0.42, color=colour,
                alpha=0.30, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.plot(r.opt, y, marker, color=colour, ms=8,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
        # A reserved column to the right, so the numbers line up and cannot
        # collide with the 0.5 rule or run off the axes.
        ax.annotate(f"{r.frac*100:.0f}%", xy=(1.0, y), xytext=(6, 0),
                    textcoords="offset points", annotation_clip=False,
                    va="center", fontsize=9, color=INK2)

    ax.annotate("window\nwidth", xy=(1.0, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(6, 1), textcoords="offset points",
                annotation_clip=False, va="bottom", fontsize=8, color=MUTED)

    ax.axvline(0.5, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=3)

    # The legend names every mark in the figure. Its handles are neutral grey:
    # colour and marker shape already tie each bar to its loss through the row
    # label, and repeating them here would suggest the legend is about one loss.
    handles = [
        Patch(facecolor="#9a9892", alpha=0.35, edgecolor="none",
              label="Operating window"),
        Line2D([], [], color="#4a4945", marker="o", ls="none", ms=6.5,
               markeredgecolor=SURFACE, markeredgewidth=1.0,
               label="Best threshold"),
        Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 3)),
               label="Default (0.5)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.27),
              ncol=3, fontsize=7.8, handletextpad=0.4, columnspacing=1.0,
              borderaxespad=0.0)

    ax.set_yticks(ys)
    ax.set_yticklabels(ORDER, fontsize=9.5, color=INK2)
    ax.set_xlabel("Decision threshold", fontsize=9, labelpad=5)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylim(-0.5, len(ORDER) - 0.4)
    ax.grid(axis="x", alpha=0.75)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    return _save(fig, stem, quiet)


# ------------------------------------------------ FIG 4  the ranking

def fig_ranking(ranking, stem="paper_fig4_ranking", quiet=False):
    """All twelve models against all five criteria.

    Include at width = \\columnwidth.

    A grid of magnitudes is a heatmap, so the colour job is sequential: one
    hue, light to dark, with the numbers printed in each cell so the figure
    doubles as the table it replaces. No colour bar: the cells carry their
    own values, and the scale is stated in the caption.
    """
    crit = ["accuracy", "ceiling", "robustness", "calibration", "transfer"]
    nice = ["FOM @0.5", "FOM best", "window", "calibration", "transfer"]

    d = ranking.sort_values("composite", ascending=False)
    mat = d[[f"n_{c}" for c in crit]].to_numpy()
    labels = [f"{r.config.replace('_', ' ')} · {r.loss}"
              for r in d.itertuples()]

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "seq", ["#eaf2fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

    fig, ax = plt.subplots(figsize=(COL_W, 4.3))
    # Margins are fixed rather than left to the layout engine. The row labels,
    # the composite column and the column titles all sit outside the axes, and
    # with automatic margins they pushed the saved figure to 4.75 in, which the
    # paper then shrinks to the column width, taking 8 pt text down to 5.7 pt.
    # Left margin = row labels (about 1.45 in); right margin = composite column.
    fig.subplots_adjust(left=0.425, right=0.885, top=0.835, bottom=0.07)
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.8,
                    color="#ffffff" if v > 0.55 else INK)

    # Column titles go on top and are vertical: five titles do not fit side by
    # side over 0.33 inch wide cells at a readable size, and a vertical title
    # costs height, which is free, instead of width, which is not.
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(nice)))
    ax.set_xticklabels(nice, rotation=90, ha="center", va="bottom",
                       fontsize=8, color=INK2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.8, color=INK2)
    ax.set_xticks(np.arange(-.5, len(nice), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)

    comp = d.composite.to_numpy()
    for i, c in enumerate(comp):
        edge = i in (0, len(comp) - 1)
        ax.annotate(f"{c:.2f}", (len(nice) - 0.42, i), annotation_clip=False,
                    va="center", fontsize=8,
                    color=INK if edge else INK2,
                    fontweight="bold" if edge else "normal")
    ax.annotate("composite", xy=(len(nice) - 0.18, -0.5),
                xytext=(0, 3), textcoords="offset points",
                annotation_clip=False, rotation=90,
                ha="center", va="bottom", fontsize=8, color=INK2)

    ax.annotate("darker = better, 1 = best of the twelve", xy=(0, 0),
                xycoords=("axes fraction", "axes fraction"),
                xytext=(0, -5), textcoords="offset points",
                annotation_clip=False, ha="left", va="top", fontsize=7.8,
                color=MUTED)

    return _save(fig, stem, quiet)


# -------------------------------------------------------------------- MAIN

def main():
    _setup()
    images = MD.available_images()
    per_model = pd.read_csv(os.path.join(config.RESULTS, "exp1_per_model.csv"))
    threshold = pd.read_csv(os.path.join(config.RESULTS, "exp2_threshold.csv"))
    ranking = pd.read_csv(os.path.join(config.RESULTS, "exp5_ranking.csv"))

    print("Building paper figures ...")
    for fig in (fig_separation(per_model), fig_reliability(images),
                fig_threshold(threshold), fig_ranking(ranking)):
        plt.close(fig)          # script mode: nothing is going to display them
    print(f"\nFigures -> {config.FIGURES}")


if __name__ == "__main__":
    main()
