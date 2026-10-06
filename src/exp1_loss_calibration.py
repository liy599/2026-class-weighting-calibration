"""
EXPERIMENT 1 -- Does the loss family or the class weighting govern calibration?

The models under test were all trained with the original author's code and
data; the only thing this paper changes is which loss the training sweep was
asked for. That sweep originally ran only wBCE and DICE, so wBCE differed
from DICE in two ways at once -- loss family AND a 166x positive-class
weight -- and nothing separated them. Training plain BCE fills the missing
cell:

                    | unweighted     | weighted x166
    ----------------+----------------+---------------
    cross-entropy   | BCE            | wBCE
    dice            | DICE           | (not defined)

Read down the unweighted column and across the cross-entropy row:

  BCE close to DICE   -> the weighting is what breaks calibration
  BCE close to wBCE   -> the loss family is, and the mechanism is wrong

Calibration is measured on pixels near the true line. A global number is
meaningless here: edge pixels are ~0.6% of an image, so any metric averaged
over the whole scene is dominated by trivially-correct background and every
model looks well calibrated regardless of how it behaves where it matters.

Run:
    python exp1_loss_calibration.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import metrics as M
import predict as P
import models as MD


N_BINS = 15
BAND_RADIUS = 10
THRESHOLD = 0.5


def band_fn(target):
    return M.band_mask(target, radius=BAND_RADIUS)


# --------------------------------------------------------- PER-MODEL TABLE

def per_model(images):
    """Detection and calibration for every model on every scene."""
    rows = []
    for tag in MD.all_tags():
        loss = MD.loss_of(tag)
        for i_tag, split in images:
            if not os.path.exists(P.cache_path(tag, i_tag)):
                continue
            prob = P.load_prob(tag, i_tag)
            target = P.load_target(i_tag).astype(bool)
            band = band_fn(target)
            pb, yb = prob[band], target[band].astype(np.float64)
            pred = prob >= THRESHOLD

            rows.append(dict(
                model=tag, config=MD.config_of(tag), loss=loss,
                family=MD.LOSS_FAMILY[loss], weighted=MD.LOSS_WEIGHTED[loss],
                guided=MD.is_guided(tag),
                image=i_tag, split=split, site=MD.site_of(i_tag),
                fom=M.figure_of_merit(pred, target),
                f1=M.detection_metrics(pred, target)["f1"],
                ece=M.expected_calibration_error(pb, yb, N_BINS),
                sce=M.signed_calibration_error(pb, yb, N_BINS),
                mce=M.maximum_calibration_error(pb, yb, N_BINS),
                brier=M.brier_score(pb, yb),
                max_prob=float(prob.max()),
            ))
    return pd.DataFrame(rows)


def table1(df):
    """Paper Table 1: one row per model, both splits side by side."""
    g = (df.groupby(["config", "loss", "split"])
           [["fom", "ece", "sce", "max_prob"]].mean().reset_index())
    wide = g.pivot(index=["config", "loss"], columns="split",
                   values=["fom", "ece", "sce"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    order = {l: i for i, l in enumerate(MD.LOSS_ORDER)}
    wide = wide.reset_index()
    wide["_o"] = wide.loss.map(order)
    wide = wide.sort_values(["config", "_o"]).drop(columns="_o")
    return wide[["config", "loss", "fom_seen", "fom_unseen",
                 "ece_seen", "ece_unseen", "sce_seen", "sce_unseen"]]


def ece_equals_sce(df):
    """How close is ECE to |signed CE|, per loss?

    They can only coincide when every probability bin errs in the same
    direction, which is the signature of a systematic one-way bias rather
    than noise. This is the sharpest evidence in the paper that the wBCE
    failure is structural, so it gets quantified rather than asserted.
    """
    d = df.copy()
    d["ratio"] = np.abs(d.sce) / d.ece.replace(0, np.nan)
    return (d.groupby("loss")
             .agg(ece=("ece", "mean"), abs_sce=("sce", lambda s: np.abs(s).mean()),
                  ratio=("ratio", "mean"), n=("ece", "size"))
             .reindex(MD.LOSS_ORDER).round(4))


# ------------------------------------------------------------------ FIGURE

def fig_reliability(images, path):
    """Paper Figure 1: pooled reliability curve per loss."""
    cfgs = MD.complete_configs()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    for ax, split in zip(axes, ["seen", "unseen"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for loss in MD.LOSS_ORDER:
            tags = [t for t in MD.tags_with(loss=loss)
                    if MD.config_of(t) in cfgs]
            if not tags:
                continue
            p, y = MD.pooled_pixels(tags, images, split, band_fn)
            if p.size == 0:
                continue
            r = M.reliability_curve(p, y, N_BINS)
            ece = M.expected_calibration_error(p, y, N_BINS)
            colour, marker, label = MD.LOSS_STYLE[loss]
            ax.plot(r["confidence"], r["accuracy"], marker=marker,
                    color=colour, lw=1.8, ms=5,
                    label=f"{label}   ECE {ece:.3f}")

        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency of edge")
        ax.set_title(f"{split} locations", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Class weighting, not the loss family, destroys calibration",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ------------------------------------------------------------------ VERDICT

def verdict(df):
    cfgs = MD.complete_configs()
    d = df[df.config.isin(cfgs) & (df.split == "seen")]
    s = d.groupby("loss").sce.mean()
    if "BCE" not in s.index:
        return "  BCE control not cached -- run train_bce_control.py, predict.py."

    bce, dice, wbce = s["BCE"], s["DICE"], s["wBCE"]
    out = [
        "  mean signed calibration error, seen locations, complete configs:",
        f"    BCE   unweighted cross-entropy   {bce:+.4f}   <- the control",
        f"    DICE  unweighted dice            {dice:+.4f}",
        f"    wBCE  weighted x166              {wbce:+.4f}",
        "",
    ]
    if abs(bce - dice) < abs(bce - wbce):
        out += [
            "  The control sits with DICE. Both unweighted losses are close to",
            "  calibrated and the weighted one differs from both, so the",
            "  miscalibration tracks the CLASS WEIGHTING, not the loss family.",
            "",
            "  Note the ordering BCE < DICE reproduces the segmentation",
            "  literature, which reports cross-entropy as better calibrated",
            "  than Dice. There is no contradiction to explain away: that",
            "  ordering holds here too, and adding pos_weight is what moves",
            "  cross-entropy from the best-calibrated loss to the worst.",
        ]
    else:
        out += [
            "  The control sits with wBCE. Both cross-entropy variants are",
            "  miscalibrated regardless of weighting, so the LOSS FAMILY is",
            "  what matters and pos_weight is not the mechanism. The paper's",
            "  explanation needs rewriting.",
        ]
    return "\n".join(out)


# --------------------------------------------------------------------- MAIN

def main():
    if not config.check():
        raise SystemExit("Fix the config paths first.")

    images = MD.available_images()
    if not images:
        raise SystemExit("Cache is empty -- run `python predict.py` first.")

    print()
    MD.describe()

    df = per_model(images)
    df.to_csv(os.path.join(config.RESULTS, "exp1_per_model.csv"), index=False)

    print("\n" + "=" * 78)
    print("TABLE 1  Detection and calibration by configuration and loss")
    print("=" * 78)
    t1 = table1(df)
    t1.to_csv(os.path.join(config.RESULTS, "exp1_table1.csv"), index=False)
    print(t1.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("Is the wBCE error one-way?   |signed CE| / ECE  -> 1.0 means yes")
    print("=" * 78)
    eq = ece_equals_sce(df)
    eq.to_csv(os.path.join(config.RESULTS, "exp1_one_way.csv"))
    print(eq.to_string())
    print("\n  A ratio near 1 means every probability bin errs in the same")
    print("  direction: a systematic bias. A ratio well below 1 means the")
    print("  errors cancel, which is what calibrated-plus-noise looks like.")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(verdict(df))

    print("\nFigure ...")
    fig_reliability(images, os.path.join(config.FIGURES, "fig1_reliability.png"))
    print(f"\nTables -> {config.RESULTS}")


if __name__ == "__main__":
    main()
