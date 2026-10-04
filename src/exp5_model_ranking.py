"""
EXPERIMENT 5 -- Which model is actually best, judged on more than accuracy?

Every comparison so far isolated one factor. This one puts all the models
side by side on the criteria a practitioner would weigh, and says plainly
which to use and which to avoid.

Five criteria, chosen because each can independently make a model unusable:

  accuracy      FOM at the default 0.5 threshold -- what you get out of the
                box, which is what most people actually get
  ceiling       FOM at the model's own best threshold -- what it could do
                if you could tune, which bounds the above
  robustness    the fraction of the threshold range where FOM stays within
                0.02 of its own best. A model with a narrow window must be
                tuned per site, and tuning needs labels from that site
  calibration   |signed calibration error| near the line -- whether the
                stated confidence can be believed
  transfer      the drop in FOM from seen to unseen coastline -- whether
                any of the above survives a beach it was not trained on

Scores are min-max normalised across models so criteria measured in
different units can be combined, then averaged with equal weight. Equal
weighting is a choice, not a fact; `WEIGHTS` makes it explicit and easy to
argue with. The per-criterion ranks are reported alongside the composite so
a reader who disagrees with the weighting can still use the table.

Run:
    python exp5_model_ranking.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import models as MD


#: Equal weight by default. Raise `robustness` if you care about shipping
#: without per-site tuning; raise `calibration` if the probabilities feed a
#: downstream uncertainty calculation.
WEIGHTS = {
    "accuracy":    1.0,
    "ceiling":     1.0,
    "robustness":  1.0,
    "calibration": 1.0,
    "transfer":    1.0,
}

CRITERIA = list(WEIGHTS)


def load():
    r = config.RESULTS
    per_model = pd.read_csv(os.path.join(r, "exp1_per_model.csv"))
    thresh = pd.read_csv(os.path.join(r, "exp2_threshold.csv"))
    return per_model, thresh


def build(per_model, thresh):
    """One row per model, five criteria, all oriented so higher is better."""
    cal = (per_model.groupby(["model", "split"]).sce.mean()
           .abs().unstack())
    fom = thresh.pivot(index="model", columns="split",
                       values=["fom_at_05", "fom_best", "window_frac"])
    fom.columns = [f"{a}_{b}" for a, b in fom.columns]

    df = fom.join(cal.rename(columns={"seen": "sce_seen",
                                      "unseen": "sce_unseen"}))
    df = df.reset_index()
    df["loss"] = df.model.map(MD.loss_of)
    df["config"] = df.model.map(MD.config_of)
    df["guided"] = df.model.map(MD.is_guided)

    # Raw criteria. Averaged over splits where both matter, so a model that
    # only works on familiar coastline cannot win on the seen column alone.
    df["accuracy"] = df[["fom_at_05_seen", "fom_at_05_unseen"]].mean(axis=1)
    df["ceiling"] = df[["fom_best_seen", "fom_best_unseen"]].mean(axis=1)
    df["robustness"] = df[["window_frac_seen", "window_frac_unseen"]].mean(axis=1)
    # Lower is better for these two, so negate before normalising.
    df["calibration"] = -df[["sce_seen", "sce_unseen"]].mean(axis=1)
    df["transfer"] = -(df.fom_best_seen - df.fom_best_unseen).abs()
    return df


def normalise(df):
    out = df.copy()
    for c in CRITERIA:
        lo, hi = out[c].min(), out[c].max()
        out[f"n_{c}"] = 0.5 if hi == lo else (out[c] - lo) / (hi - lo)
        out[f"rank_{c}"] = out[c].rank(ascending=False, method="min").astype(int)

    w = np.array([WEIGHTS[c] for c in CRITERIA], dtype=float)
    w = w / w.sum()
    out["composite"] = (out[[f"n_{c}" for c in CRITERIA]].to_numpy() * w).sum(axis=1)
    out["rank"] = out.composite.rank(ascending=False, method="min").astype(int)
    return out.sort_values("composite", ascending=False)


# ------------------------------------------------------------------ FIGURE

def fig_ranking(df, path):
    """Stacked contribution of each criterion to the composite score."""
    d = df.sort_values("composite")
    labels = [f"{MD.config_of(m)}\n{MD.loss_of(m)}" for m in d.model]
    w = np.array([WEIGHTS[c] for c in CRITERIA]); w = w / w.sum()

    palette = ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#b2182b"]
    fig, ax = plt.subplots(figsize=(9.5, 0.46 * len(d) + 2.0))

    left = np.zeros(len(d))
    for c, colour, wi in zip(CRITERIA, palette, w):
        vals = d[f"n_{c}"].to_numpy() * wi
        ax.barh(labels, vals, left=left, color=colour, label=c,
                edgecolor="white", linewidth=0.6)
        left += vals

    ax.set_xlabel("Composite score (equal-weighted, normalised)")
    ax.set_xlim(0, max(1.0, left.max() * 1.02))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=len(CRITERIA), loc="lower center",
              bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("Which model to use, on five criteria at once", y=1.0,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ----------------------------------------------------------------- REPORT

def report(df):
    print("\n" + "=" * 100)
    print("TABLE 5  Multi-factor comparison  (higher is better in every column)")
    print("=" * 100)
    show = df[["rank", "config", "loss", "accuracy", "ceiling", "robustness",
               "calibration", "transfer", "composite"]].copy()
    show["calibration"] = -show["calibration"]      # print as |error|
    show["transfer"] = -show["transfer"]            # print as FOM drop
    show = show.rename(columns={"accuracy": "fom@0.5", "ceiling": "fom_best",
                                "robustness": "window",
                                "calibration": "|cal_err|",
                                "transfer": "fom_drop"})
    print(show.round(4).to_string(index=False))
    print("\n  |cal_err| and fom_drop are printed as raw magnitudes, so for")
    print("  those two columns smaller is better; they were negated before")
    print("  scoring so that the composite is consistently higher-is-better.")

    best, worst = df.iloc[0], df.iloc[-1]
    print("\n" + "=" * 100)
    print("BEST AND WORST")
    print("=" * 100)
    print(f"  BEST   {best.config}_{best.loss}   composite {best.composite:.3f}")
    print(f"         FOM@0.5 {best.accuracy:.3f}   ceiling {best.ceiling:.3f}   "
          f"window {best.robustness:.3f}   |cal err| {-best.calibration:.4f}")
    print(f"  WORST  {worst.config}_{worst.loss}   composite {worst.composite:.3f}")
    print(f"         FOM@0.5 {worst.accuracy:.3f}   ceiling {worst.ceiling:.3f}   "
          f"window {worst.robustness:.3f}   |cal err| {-worst.calibration:.4f}")

    print("\n" + "=" * 100)
    print("Who wins each criterion on its own?")
    print("=" * 100)
    for c in CRITERIA:
        w = df.loc[df[c].idxmax()]
        l = df.loc[df[c].idxmin()]
        print(f"  {c:<12s} best {w.config}_{w.loss:<8s}   "
              f"worst {l.config}_{l.loss}")

    print("\n" + "=" * 100)
    print("Marginal effect of each factor (mean composite)")
    print("=" * 100)
    for factor, col in [("loss", "loss"), ("backbone", "config"),
                        ("guidance", "guided")]:
        if factor == "backbone":
            g = df.assign(k=df.config.str.split("_").str[0]).groupby("k")
        elif factor == "guidance":
            g = df.groupby(df.guided.map({True: "guided", False: "unguided"}))
        else:
            g = df.groupby(col)
        m = g.composite.mean().sort_values(ascending=False).round(3)
        print(f"\n  by {factor}:")
        for k, v in m.items():
            print(f"    {str(k):<14s} {v:.3f}")


def main():
    if not config.check():
        raise SystemExit("Fix the config paths first.")
    per_model, thresh = load()
    df = normalise(build(per_model, thresh))
    df.to_csv(os.path.join(config.RESULTS, "exp5_ranking.csv"), index=False)
    report(df)
    print("\nFigure ...")
    fig_ranking(df, os.path.join(config.FIGURES, "fig5_ranking.png"))
    print(f"\nTables -> {config.RESULTS}")


if __name__ == "__main__":
    main()
