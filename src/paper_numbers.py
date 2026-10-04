"""
Every number quoted in the paper, computed from the results and the cache.

Why this script exists
----------------------
The experiment scripts write tables and figures, but a paper also quotes
derived numbers: a mean over four settings, a ratio between two losses, the
gap between a theoretical curve and a measured one. This script computes each
of them in one place, so that anyone can see where a number in the text comes
from, and checks it against the value printed in the manuscript. If you
retrain the models and a quoted number no longer holds, this says which.

It also writes the body rows of the two tables in the paper, so the LaTeX is
never typed by hand:

    results/paper_numbers.txt          this report
    results/paper_table1_rows.tex      Table 1, twelve models
    results/paper_table2_rows.tex      Table 2, the three losses at a glance

Needs the experiment outputs in results/ (run exp1 to exp5 first) and the
probability cache (a few numbers are computed from the pixels themselves).

Run:
    python paper_numbers.py
"""

import glob
import os

import numpy as np
import pandas as pd

import config
import metrics as M
import models as MD
import predict as P


R = config.RESULTS
PX_BAND = 10
N_BINS = 15
W = 166.0

LINES, CHECKS = [], []


def say(s=""):
    print(s)
    LINES.append(s)


def head(title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


def quoted(label, value, paper, tol):
    """A number that the paper prints, with the tolerance of its rounding."""
    ok = abs(value - paper) <= tol
    CHECKS.append(ok)
    say(f"  {'OK  ' if ok else 'DIFF'}  {label:<56s} computed {value:>9.4f}   paper {paper}")


def at_least(label, value, paper):
    """A claim of the form 'more than N times'."""
    ok = value >= paper
    CHECKS.append(ok)
    say(f"  {'OK  ' if ok else 'DIFF'}  {label:<56s} computed {value:>9.2f}   paper: more than {paper}")


def minus(v):
    """A number as it is typeset in the paper: a true minus sign, no hyphen."""
    s = f"{abs(v):.4f}"
    return "\\textminus" + s if v < 0 else s


def read(name):
    return pd.read_csv(os.path.join(R, name))


def main():
    per = read("exp1_per_model.csv")
    one_way = read("exp1_one_way.csv").set_index("loss")
    t1 = read("exp1_table1.csv")
    thr = read("exp2_threshold.csv")
    rec = read("exp3_recalibration.csv")
    sep = read("exp4_separation.csv")
    sens = read("exp4_sensitivity.csv")
    rank = read("exp5_ranking.csv")
    images = MD.available_images()
    band = lambda t: M.band_mask(t, radius=PX_BAND)

    # ----------------------------------------------------------- Section III
    head("III  Data")
    seen = [i for i, s in images if s == "seen"]
    unseen = [i for i, s in images if s == "unseen"]
    quoted("test images, seen", len(seen), 9, 0)
    quoted("test images, unseen", len(unseen), 16, 0)
    quoted("beaches in the seen set", len({MD.site_of(i) for i in seen}), 4, 0)

    train = sorted(glob.glob(os.path.join(config.DATA, "training", "*.npy")))
    if train:
        fr = np.mean([np.load(p, mmap_mode="r")[-1].astype(bool).mean() for p in train])
        quoted("training crops", len(train), 8626, 0)
        quoted("edge pixels in a training crop (percent)", 100 * fr, 0.6, 0.1)
        quoted("class weight = 1 / prevalence", 1 / fr, 166, 12)
    else:
        say("  (training data not found: skipped crop count and prevalence)")

    prev = [P.load_target(i).astype(bool).mean() for i, _ in images]
    quoted("edge pixels in a whole test scene (percent, mean)", 100 * np.mean(prev), 0.05, 0.005)

    # Pratt's figure of merit gives each predicted pixel 1 / (1 + alpha d^2)
    a = 1 / 9
    quoted("FOM score of a pixel 1 px from the true line", 1 / (1 + a * 1), 0.9, 0.005)
    quoted("FOM score of a pixel 10 px from the true line", 1 / (1 + a * 100), 0.08, 0.005)

    # ------------------------------------------------------------ Section V-A
    head("V-A  Weighting, not the type of loss")
    g = per.groupby(["config", "loss", "split"]).sce.mean().reset_index()
    sce = g.groupby(["loss", "split"]).sce.mean().unstack()
    for loss, p_seen in (("BCE", -0.0013), ("DICE", 0.0142), ("wBCE", 0.1505)):
        quoted(f"signed error, seen, {loss} (mean of 4 settings)", sce.loc[loss, "seen"], p_seen, 0.00005)
    for loss, p_un in (("BCE", -0.0167), ("DICE", -0.0067), ("wBCE", 0.1127)):
        quoted(f"signed error, unseen, {loss}", sce.loc[loss, "unseen"], p_un, 0.00005)
    quoted("wBCE seen vs unseen (paper: 0.151 vs 0.113)", sce.loc["wBCE", "unseen"], 0.113, 0.0005)

    gap = (g[(g.split == "seen")].pivot(index="config", columns="loss", values="sce"))
    gap = (gap["wBCE"] - gap["BCE"])
    quoted("largest wBCE minus BCE gap, seen", gap.max(), 0.209, 0.0005)
    say(f"        found in the {gap.idxmax().replace('_', ', ')} setting; smallest gap {gap.min():.3f}")

    quoted("signed error / ECE, wBCE", one_way.loc["wBCE", "ratio"], 0.91, 0.005)
    quoted("signed error / ECE, DICE", one_way.loc["DICE", "ratio"], 0.37, 0.005)
    quoted("signed error / ECE, BCE", one_way.loc["BCE", "ratio"], 0.64, 0.005)

    # pooled over all pixels of the models that share a loss (as in Fig. 2)
    pool, rel = {}, {}
    for loss in MD.LOSS_ORDER:
        p, y = MD.pooled_pixels(MD.tags_with(loss=loss), images, "seen", band)
        pool[loss] = (p, y)
        rel[loss] = M.reliability_curve(p, y, N_BINS)
    ece = {l: M.expected_calibration_error(*pool[l], N_BINS) for l in MD.LOSS_ORDER}
    quoted("pooled ECE, BCE", ece["BCE"], 0.003, 0.0005)
    quoted("pooled ECE, DICE", ece["DICE"], 0.037, 0.0005)
    quoted("pooled ECE, wBCE", ece["wBCE"], 0.151, 0.0005)

    # ------------------------------------------------------------ Section V-B
    head("V-B  Why it happens")
    qstar = lambda p: W * p / (1 - p + W * p)
    quoted("optimum output for true chance 0.006", qstar(0.006), 0.50, 0.005)
    quoted("optimum output for true chance 0.5", qstar(0.5), 0.994, 0.0005)

    r = rel["wBCE"]
    q = r["confidence"]
    theory = q / (W + q - q * W)                      # the inverse of the optimum
    wt = r["count"] / r["count"].sum()
    resid = float(np.sum(wt * np.abs(theory - r["accuracy"])))
    quoted("average gap, theory curve vs measured (wBCE)", resid, 0.014, 0.0005)
    quoted("share of the wBCE error explained by the loss", 1 - resid / ece["wBCE"], 0.9, 0.02)

    mid = {l: float(np.median(rel[l]["count"][1:-1])) for l in MD.LOSS_ORDER}
    say(f"        median pixels in a middle bin: BCE {mid['BCE']:.0f}, "
        f"DICE {mid['DICE']:.0f}, wBCE {mid['wBCE']:.0f}")
    quoted("middle bins, BCE vs DICE (paper: about twenty times)", mid["BCE"] / mid["DICE"], 20, 3)
    quoted("middle bins, wBCE vs DICE", mid["wBCE"] / mid["DICE"], 20, 3)
    at_least("ECE of DICE relative to BCE", ece["DICE"] / ece["BCE"], 10)

    # ------------------------------------------------------------ Section V-C
    head("V-C  What the weight buys")
    by = thr.groupby("loss").agg(fom=("fom_best", "mean"), win=("window_frac", "mean"),
                                 lo=("best_threshold", "min"), hi=("best_threshold", "max"))
    for loss, v in (("BCE", 0.764), ("DICE", 0.731), ("wBCE", 0.737)):
        quoted(f"best FOM, {loss}", by.loc[loss, "fom"], v, 0.0005)
    wins = sum(d.set_index("loss").fom_best.idxmax() == "BCE"
               for _, d in thr.groupby(["config", "split"]))
    quoted("comparisons won by BCE (of 8)", wins, 7, 0)
    quoted("wBCE best threshold, lowest", by.loc["wBCE", "lo"], 0.96, 0.005)
    quoted("wBCE best threshold, highest", by.loc["wBCE", "hi"], 0.98, 0.005)
    quoted("operating window, wBCE", by.loc["wBCE", "win"], 0.08, 0.005)
    quoted("operating window, DICE", by.loc["DICE", "win"], 0.68, 0.005)
    quoted("operating window, BCE", by.loc["BCE", "win"], 0.19, 0.005)
    at_least("DICE window relative to wBCE window", by.loc["DICE", "win"] / by.loc["wBCE", "win"], 8)
    quoted("BCE best threshold, lowest", by.loc["BCE", "lo"], 0.06, 0.005)
    quoted("BCE best threshold, highest", by.loc["BCE", "hi"], 0.42, 0.005)
    quoted("DICE best threshold, lowest", by.loc["DICE", "lo"], 0.02, 0.005)
    quoted("DICE best threshold, highest", by.loc["DICE", "hi"], 0.98, 0.005)

    # ------------------------------------------------------------ Section V-D
    head("V-D  Robustness and repair after training")
    quoted("measurement settings tested", len(sep), 48, 0)
    quoted("settings where wBCE is the most overconfident", int((sep.gap > 0).sum()), 48, 0)
    quoted("smallest gap over all settings", sep.gap.min(), 0.063, 0.0005)
    s5 = sens[(sens.radius == 5) & (sens.split == "seen")].groupby("loss").sce.mean()
    quoted("5 px band, wBCE", s5["wBCE"], 0.276, 0.0005)
    quoted("5 px band, DICE", s5["DICE"], 0.028, 0.0005)

    w = rec[(rec.loss == "wBCE") & (rec.fit_on == "seen") & (rec.eval_on == "unseen")]
    w = w.groupby("calibrator").ece.mean()
    quoted("wBCE ECE on the unseen beach, uncorrected", w["none"], 0.120, 0.0006)
    quoted("  after Platt scaling", w["platt"], 0.016, 0.0006)
    quoted("  after temperature scaling", w["temperature"], 0.140, 0.0006)

    # ------------------------------------------------------------ Section V-E
    head("V-E  Comparing all twelve models")
    rk = rank.sort_values("rank")
    quoted(f"best composite ({rk.iloc[0].config}, {rk.iloc[0].loss})", rk.iloc[0].composite, 0.98, 0.005)
    quoted(f"worst composite ({rk.iloc[-1].config}, {rk.iloc[-1].loss})", rk.iloc[-1].composite, 0.08, 0.005)
    tg = rank[(rank.config == "trainable_guided") & (rank.loss == "BCE")].iloc[0]
    quoted("operating window, trainable guided BCE", tg.robustness, 0.30, 0.005)
    best_t = rank.loc[rank.transfer.idxmax()]
    quoted(f"smallest seen to unseen drop ({best_t.config}, {best_t.loss})", -best_t.transfer, 0.004, 0.0006)
    gl = rank.groupby("guided").composite.mean()
    ll = rank.groupby("loss").composite.mean()
    quoted("guided minus unguided, mean composite", gl.max() - gl.min(), 0.43, 0.005)
    quoted("best loss minus worst loss, mean composite", ll.max() - ll.min(), 0.28, 0.005)

    # ------------------------------------------------------------- the tables
    head("Table rows for the paper")
    cfgs = ["trainable_guided", "frozen_guided", "trainable_unguided", "frozen_unguided"]
    label = {"trainable_guided": "Trainable, guided", "frozen_guided": "Frozen, guided",
             "trainable_unguided": "Trainable, unguided", "frozen_unguided": "Frozen, unguided"}
    rows1 = []
    for i, cfg in enumerate(cfgs):
        for j, loss in enumerate(MD.LOSS_ORDER):
            x = t1[(t1.config == cfg) & (t1.loss == loss)].iloc[0]
            rows1.append(f"{label[cfg] if j == 0 else ''} & {loss} & {x.fom_seen:.3f} & "
                         f"{x.fom_unseen:.3f} & {minus(x.sce_seen)} & {minus(x.sce_unseen)} \\\\")
        if i < len(cfgs) - 1:
            rows1.append("\\midrule")
    rows2 = [f"{l} & {minus(sce.loc[l, 'seen'])} & {ece[l]:.3f} & {by.loc[l, 'fom']:.3f} & "
             f"{by.loc[l, 'win']:.2f} & {by.loc[l, 'lo']:.2f} to {by.loc[l, 'hi']:.2f} \\\\"
             for l in MD.LOSS_ORDER]
    for name, rows in (("paper_table1_rows.tex", rows1), ("paper_table2_rows.tex", rows2)):
        with open(os.path.join(R, name), "w", encoding="ascii", newline="\n") as f:
            f.write("\n".join(rows) + "\n")
        say(f"  wrote results/{name}  ({len(rows)} lines)")

    # ---------------------------------------------------------------- summary
    head("Summary")
    n_ok = sum(CHECKS)
    say(f"  {n_ok} of {len(CHECKS)} quoted numbers match the manuscript")
    if n_ok != len(CHECKS):
        say("  Lines marked DIFF no longer match: retrained models, or a changed "
            "experiment, or the manuscript needs updating.")
    with open(os.path.join(R, "paper_numbers.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    say("  wrote results/paper_numbers.txt")


if __name__ == "__main__":
    if not config.check():
        raise SystemExit("Fix the config paths first.")
    main()
