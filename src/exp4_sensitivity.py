"""
EXPERIMENT 4 -- Does the finding survive the choices we made to measure it?

Expected calibration error is known to depend on how the probability range
is partitioned: too few bins hide the error, too many make it noisy, and
equal-width bins behave differently from equal-count ones when the
probabilities pile up near zero -- which, on 0.6% positives, they do.

The paper's central claim is a *separation* between weighted and unweighted
losses, so what has to be robust is the separation, not any single number.
This sweeps the three measurement choices we made and checks that the gap
never closes:

  n_bins        10, 15, 20, 30
  strategy      uniform (equal width) or quantile (equal count)
  band radius   5, 10, 20 px around the true line

The band radius is the one with real content rather than bookkeeping. A
narrow band is the hardest region -- pixels the model must decide between
-- while a wide one dilutes towards easy background, so calibration error
should shrink as the radius grows. If the wBCE/unweighted separation
survived only at one radius it would be an artefact of where we chose to
look.

Run:
    python exp4_sensitivity.py
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import metrics as M
import models as MD


N_BINS = [10, 15, 20, 30]
STRATEGIES = ["uniform", "quantile"]
RADII = [5, 10, 20]
SUBSAMPLE = 150_000


def run(images):
    cfgs = MD.complete_configs()
    rows = []

    for radius in RADII:
        band = lambda t, r=radius: M.band_mask(t, radius=r)
        # Pool once per (loss, split, radius); the binning sweep is then free
        pools = {}
        for loss in MD.LOSS_ORDER:
            tags = [t for t in MD.tags_with(loss=loss)
                    if MD.config_of(t) in cfgs]
            for split in ("seen", "unseen"):
                pools[(loss, split)] = MD.pooled_pixels(
                    tags, images, split, band, subsample=SUBSAMPLE)

        for (loss, split), (p, y) in pools.items():
            if p.size == 0:
                continue
            for n_bins, strategy in itertools.product(N_BINS, STRATEGIES):
                rows.append(dict(
                    loss=loss, split=split, radius=radius,
                    n_bins=n_bins, strategy=strategy,
                    ece=M.expected_calibration_error(p, y, n_bins, strategy),
                    sce=M.signed_calibration_error(p, y, n_bins, strategy),
                    mce=M.maximum_calibration_error(p, y, n_bins, strategy),
                ))
        print(f"  radius {radius:>2d} px done")

    return pd.DataFrame(rows)


def separation(df):
    """The gap between the worst unweighted loss and weighted wBCE.

    Positive everywhere means the separation never closes: no setting of the
    measurement knobs makes an unweighted loss look as miscalibrated as the
    weighted one.
    """
    rows = []
    keys = ["split", "radius", "n_bins", "strategy"]
    for k, g in df.groupby(keys):
        s = g.set_index("loss").sce
        if not set(MD.LOSS_ORDER) <= set(s.index):
            continue
        unweighted_worst = max(s["BCE"], s["DICE"])
        rows.append(dict(zip(keys, k), bce=s["BCE"], dice=s["DICE"],
                         wbce=s["wBCE"], gap=s["wBCE"] - unweighted_worst))
    return pd.DataFrame(rows)


def fig_sensitivity(df, path):
    """Paper Figure 4: the separation under every measurement choice."""
    fig, axes = plt.subplots(1, len(RADII), figsize=(4.0 * len(RADII), 3.8),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, radius in zip(axes, RADII):
        d = df[(df.radius == radius) & (df.split == "seen")]
        for loss in MD.LOSS_ORDER:
            colour, marker, label = MD.LOSS_STYLE[loss]
            for strategy, ls in [("uniform", "-"), ("quantile", "--")]:
                s = d[(d.loss == loss) & (d.strategy == strategy)]
                s = s.sort_values("n_bins")
                if s.empty:
                    continue
                ax.plot(s.n_bins, s.sce, ls, marker=marker, color=colour,
                        lw=1.6, ms=4.5,
                        label=label if strategy == "uniform" else None)
        ax.axhline(0, color="#999", lw=1, ls=":")
        ax.set_xlabel("Number of bins")
        ax.set_title(f"band radius {radius} px", fontsize=10)
        ax.set_xticks(N_BINS)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Signed calibration error")
    axes[0].legend(fontsize=8, loc="center left")

    fig.suptitle("The separation holds under every binning and band choice "
                 "(solid: equal width, dashed: equal count)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def summarise(df, sep):
    print("\n" + "=" * 78)
    print("Signed calibration error by loss, across every measurement choice")
    print("=" * 78)
    t = (df.groupby(["loss", "split"]).sce
           .agg(["min", "max", "mean"])
           .reindex([(l, s) for l in MD.LOSS_ORDER
                     for s in ("seen", "unseen")]).round(4))
    print(t.to_string())

    print("\n" + "=" * 78)
    print("TABLE 4  Does the separation ever close?")
    print("=" * 78)
    print(f"  settings tested            : {len(sep)}")
    print(f"  settings where wBCE is the most overconfident : "
          f"{int((sep.gap > 0).sum())} / {len(sep)}")
    print(f"  smallest gap               : {sep.gap.min():+.4f}")
    print(f"  largest gap                : {sep.gap.max():+.4f}")

    worst = sep.loc[sep.gap.idxmin()]
    print(f"\n  narrowest case: {worst.split}, radius {int(worst.radius)} px, "
          f"{int(worst.n_bins)} {worst.strategy} bins")
    print(f"    BCE {worst.bce:+.4f}   DICE {worst.dice:+.4f}   "
          f"wBCE {worst.wbce:+.4f}")

    if (sep.gap > 0).all():
        print("\n  The separation never closes. The finding is a property of")
        print("  the models, not of how we chose to measure them.")
    else:
        bad = sep[sep.gap <= 0]
        print(f"\n  WARNING: the separation closes in {len(bad)} settings.")
        print(bad.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("Effect of band radius (mean over binning choices, seen)")
    print("=" * 78)
    r = (df[df.split == "seen"].groupby(["radius", "loss"]).sce.mean()
           .unstack()[MD.LOSS_ORDER].round(4))
    print(r.to_string())
    print("\n  Error shrinking with radius is expected: a wider band mixes in")
    print("  easy background. What matters is that the ordering is unchanged.")


def main():
    if not config.check():
        raise SystemExit("Fix the config paths first.")
    images = MD.available_images()
    if not images:
        raise SystemExit("Cache is empty -- run `python predict.py` first.")

    n = len(N_BINS) * len(STRATEGIES) * len(RADII)
    print(f"\nSweeping {n} measurement settings per loss and split")
    df = run(images)
    df.to_csv(os.path.join(config.RESULTS, "exp4_sensitivity.csv"), index=False)

    sep = separation(df)
    sep.to_csv(os.path.join(config.RESULTS, "exp4_separation.csv"), index=False)

    summarise(df, sep)

    print("\nFigure ...")
    fig_sensitivity(df, os.path.join(config.FIGURES, "fig4_sensitivity.png"))
    print(f"\nTables -> {config.RESULTS}")


if __name__ == "__main__":
    main()
