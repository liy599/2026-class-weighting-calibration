"""
EXPERIMENT 2 -- What does the class weight actually buy?

`pos_weight = 166` exists to stop the network from predicting "no edge"
everywhere, which on 0.6% positives would be 99.4% correct and useless.
The assumption behind it is that without the weight the model cannot find
the line at all.

That assumption is testable, and it conflates two different things:

  can the model rank edge pixels above non-edge pixels?   (ability)
  does p = 0.5 happen to be where the cut belongs?        (threshold)

An unweighted model trained on 0.6% positives outputs low probabilities
everywhere, so a fixed 0.5 cut predicts almost nothing -- which looks like
failure but is only a mis-placed threshold. Sweeping the threshold
separates the two, and the difference between FOM at 0.5 and FOM at the
best threshold is exactly what the weight is buying.

Caveat, stated here because it constrains the claim: the best threshold is
chosen on the same data it is evaluated on. It is an upper bound on what
threshold tuning could achieve, not a deployable recipe. The paper must say
so -- the honest claim is about what the weight does and does not buy,
not that one should ship a tuned threshold.

Run:
    python exp2_threshold.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import metrics as M
import predict as P
import models as MD


SWEEP = np.round(np.arange(0.02, 0.991, 0.02), 3)
THRESHOLD = 0.5


def operating_window(foms, tol=0.02):
    """Width of the threshold range where FOM stays within `tol` of its best.

    This is what a practitioner actually cares about. A model whose accuracy
    is flat across most of the range can be shipped at any sensible cut; one
    with a narrow peak has to be tuned per deployment, and tuning needs
    labels from the deployment site, which is exactly what nobody has.
    """
    ok = foms >= foms.max() - tol
    return float(ok.sum()) / len(foms), float(SWEEP[ok].min()), float(SWEEP[ok].max())


def sweep_model(tag, images, split):
    """Mean FOM across a split's scenes, at every threshold."""
    foms, n = np.zeros(len(SWEEP)), 0
    for i_tag, s in images:
        if s != split or not os.path.exists(P.cache_path(tag, i_tag)):
            continue
        prob = P.load_prob(tag, i_tag)
        target = P.load_target(i_tag).astype(bool)
        foms += np.array([M.figure_of_merit(prob >= t, target) for t in SWEEP])
        n += 1
    return (foms / n, n) if n else (None, 0)


def run(images):
    rows, curves = [], {}
    for tag in MD.all_tags():
        for split in ["seen", "unseen"]:
            foms, n = sweep_model(tag, images, split)
            if foms is None:
                continue
            curves[(tag, split)] = foms
            best = int(np.argmax(foms))
            at05 = float(foms[int(np.argmin(np.abs(SWEEP - THRESHOLD)))])
            frac, lo, hi = operating_window(foms)
            rows.append(dict(
                model=tag, config=MD.config_of(tag), loss=MD.loss_of(tag),
                split=split, n_scenes=n,
                fom_at_05=at05,
                fom_best=float(foms[best]),
                best_threshold=float(SWEEP[best]),
                recovered=float(foms[best]) - at05,
                window_frac=frac, window_lo=lo, window_hi=hi,
            ))
    return pd.DataFrame(rows), curves


def fig_threshold(curves, path):
    """Paper Figure 2: FOM against threshold, one panel per split."""
    cfgs = MD.complete_configs()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    for ax, split in zip(axes, ["seen", "unseen"]):
        for loss in MD.LOSS_ORDER:
            colour, _, label = MD.LOSS_STYLE[loss]
            tags = [t for t in MD.tags_with(loss=loss)
                    if MD.config_of(t) in cfgs]
            got = [curves[(t, split)] for t in tags if (t, split) in curves]
            if not got:
                continue
            mean = np.mean(got, axis=0)
            ax.plot(SWEEP, mean, "-", color=colour, lw=2, label=label)
            b = int(np.argmax(mean))
            ax.plot(SWEEP[b], mean[b], "o", color=colour, ms=7,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=5)

        ax.axvline(THRESHOLD, color="#666", ls=":", lw=1.2)
        ax.text(THRESHOLD - 0.02, 0.06, "default 0.5", fontsize=8,
                color="#666", rotation=90, va="bottom", ha="right")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Figure of merit")
        ax.set_title(f"{split} locations", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower center")

    fig.suptitle("Only DICE is insensitive to the threshold; the weight moves "
                 "the peak rather than fixing it\n(dots mark each loss's "
                 "optimum)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def summarise(df):
    cfgs = MD.complete_configs()
    d = df[df.config.isin(cfgs)]

    print("\n" + "=" * 78)
    print("TABLE 2  Accuracy at the default threshold vs at the best one")
    print("=" * 78)
    t = (d.pivot_table(index=["config", "split"], columns="loss",
                       values=["fom_at_05", "fom_best", "best_threshold"])
           .round(3))
    print(t.to_string())

    print("\n" + "=" * 78)
    print("Who wins on accuracy, once the threshold is not held at 0.5?")
    print("=" * 78)
    for split in ["seen", "unseen"]:
        for cfg in cfgs:
            s = d[(d.split == split) & (d.config == cfg)].set_index("loss")
            if not set(MD.LOSS_ORDER) <= set(s.index):
                continue
            best = s.fom_best.idxmax()
            print(f"  {cfg:<20s} {split:<7s} "
                  + "  ".join(f"{l} {s.loc[l,'fom_best']:.3f}"
                              for l in MD.LOSS_ORDER)
                  + f"   -> {best}")

    print("\n" + "=" * 78)
    print("Where each loss's optimum sits, and how wide the usable range is")
    print("=" * 78)
    st = (d.groupby("loss")
            .agg(best_min=("best_threshold", "min"),
                 best_max=("best_threshold", "max"),
                 window=("window_frac", "mean"),
                 win_lo=("window_lo", "mean"),
                 win_hi=("window_hi", "mean"))
            .reindex(MD.LOSS_ORDER).round(3))
    print(st.to_string())
    print("\n  'window' is the fraction of the threshold range over which FOM")
    print("  stays within 0.02 of its own best -- the range you could ship")
    print("  without tuning. Tuning needs labels from the deployment site,")
    print("  which is the one thing a practitioner does not have.")

    print("\n  Note that 0.5 is not the right cut for wBCE either: its optimum")
    print("  sits near the top of the range. The weight does not deliver a")
    print("  correct default threshold, it just moves where the model fails.")


def main():
    if not config.check():
        raise SystemExit("Fix the config paths first.")
    images = MD.available_images()
    if not images:
        raise SystemExit("Cache is empty -- run `python predict.py` first.")

    print(f"\nSweeping {len(SWEEP)} thresholds over {len(MD.all_tags())} models")
    df, curves = run(images)
    df.to_csv(os.path.join(config.RESULTS, "exp2_threshold.csv"), index=False)

    summarise(df)

    print("\nFigure ...")
    fig_threshold(curves, os.path.join(config.FIGURES, "fig2_threshold.png"))
    print(f"\nTables -> {config.RESULTS}")


if __name__ == "__main__":
    main()
