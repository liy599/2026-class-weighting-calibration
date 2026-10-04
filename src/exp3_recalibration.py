"""
EXPERIMENT 3 -- If you are stuck with a weighted model, can it be repaired?

Experiments 1 and 2 say what to do when training from scratch. Most readers
are not: they have a trained wBCE model and cannot afford to retrain. So the
practical question is whether the probabilities can be fixed afterwards.

The interesting part is that the popular method provably cannot work here.

  Temperature scaling maps logit z -> z/T. Since 0/T = 0 for every T, and
  logit(0.5) = 0, the point p = 0.5 is a fixed point of every temperature.
  Temperature can sharpen or flatten a curve about 0.5, but it cannot move
  it. Experiment 1 shows the wBCE curve depressed *everywhere*, including
  below 0.5, so the correction it needs is a shift.

  Platt scaling maps z -> a*z + b. The intercept b is exactly the shift
  that temperature lacks.

So the prediction is made before the experiment, not after: temperature
should fail and Platt should succeed. Confirming it turns "systematic
offset, not sharpness" from an assertion into a measurement.

Parameters are fitted on the seen beaches and applied unchanged to the
held-out beach, because a correction that needs labels from the deployment
site is of no use to anyone. Fitting on the unseen beach as well gives the
oracle, and the gap between the two is the transfer cost.

Run:
    python exp3_recalibration.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import config
import metrics as M
import models as MD


N_BINS = 15
BAND_RADIUS = 10
EPS = 1e-6
SUBSAMPLE = 200_000        # per model per scene; keeps the fits quick


def band_fn(target):
    return M.band_mask(target, radius=BAND_RADIUS)


# ------------------------------------------------------------------ LOGITS

def to_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ------------------------------------------------------------ CALIBRATORS

def fit_temperature(p, y):
    z = to_logit(p)
    r = minimize(lambda t: nll(sigmoid(z / np.exp(t[0])), y), x0=[0.0],
                 method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-6, maxiter=500))
    return {"T": float(np.exp(r.x[0]))}


def fit_platt(p, y):
    z = to_logit(p)
    r = minimize(lambda th: nll(sigmoid(th[0] * z + th[1]), y), x0=[1.0, 0.0],
                 method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-6, maxiter=2000))
    return {"a": float(r.x[0]), "b": float(r.x[1])}


CALIBRATORS = {
    "none":        (lambda p, y: {}, lambda p, q: np.asarray(p)),
    "temperature": (fit_temperature, lambda p, q: sigmoid(to_logit(p) / q["T"])),
    "platt":       (fit_platt, lambda p, q: sigmoid(q["a"] * to_logit(p) + q["b"])),
}


def score(p, y):
    return dict(
        ece=M.expected_calibration_error(p, y, N_BINS),
        sce=M.signed_calibration_error(p, y, N_BINS),
        mce=M.maximum_calibration_error(p, y, N_BINS),
        brier=M.brier_score(p, y),
        nll=nll(p, y),
    )


# --------------------------------------------------------------------- RUN

def run(images):
    rows = []
    for tag in MD.all_tags():
        pool = {s: MD.pooled_pixels([tag], images, s, band_fn,
                                    subsample=SUBSAMPLE)
                for s in ("seen", "unseen")}
        if pool["seen"][0].size == 0:
            continue

        for cal, (fit, apply) in CALIBRATORS.items():
            params = fit(*pool["seen"])
            for split in ("seen", "unseen"):
                p, y = pool[split]
                if p.size == 0:
                    continue
                rows.append(dict(
                    model=tag, config=MD.config_of(tag), loss=MD.loss_of(tag),
                    calibrator=cal, fit_on="seen", eval_on=split,
                    params=str({k: round(v, 4) for k, v in params.items()}),
                    **score(apply(p, params), y)))

            if cal != "none" and pool["unseen"][0].size:
                own = fit(*pool["unseen"])
                p, y = pool["unseen"]
                rows.append(dict(
                    model=tag, config=MD.config_of(tag), loss=MD.loss_of(tag),
                    calibrator=cal, fit_on="unseen", eval_on="unseen",
                    params=str({k: round(v, 4) for k, v in own.items()}),
                    **score(apply(p, own), y)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ FIGURE

def fig_recalibration(images, df, path):
    """Paper Figure 3: what each correction does to the wBCE curve."""
    tags = [t for t in MD.tags_with(loss="wBCE")
            if MD.config_of(t) in MD.complete_configs()]
    if not tags:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, split in zip(axes, ["seen", "unseen"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        p_all, y_all = MD.pooled_pixels(tags, images, split, band_fn)
        p_fit, y_fit = MD.pooled_pixels(tags, images, "seen", band_fn)
        if p_all.size == 0:
            continue
        for cal, colour, marker in [("none", "#b2182b", "s"),
                                    ("temperature", "#f4a582", "^"),
                                    ("platt", "#2166ac", "o")]:
            fit, apply = CALIBRATORS[cal]
            params = fit(p_fit, y_fit)
            p_cal = apply(p_all, params)
            r = M.reliability_curve(p_cal, y_all, N_BINS)
            ece = M.expected_calibration_error(p_cal, y_all, N_BINS)
            label = {"none": "uncorrected", "temperature": "temperature",
                     "platt": "Platt"}[cal]
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

    fig.suptitle("Temperature cannot shift a curve; Platt can "
                 "(wBCE models, fitted on seen locations)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def summarise(df):
    print("\n" + "=" * 78)
    print("TABLE 3  ECE after correction, fitted on seen locations")
    print("=" * 78)
    d = df[df.fit_on == "seen"]
    t = d.pivot_table(index=["loss", "eval_on"], columns="calibrator",
                      values="ece").reindex(
        [(l, s) for l in MD.LOSS_ORDER for s in ("seen", "unseen")])
    print(t[["none", "temperature", "platt"]].round(4).to_string())

    print("\n  Temperature scaling is expected to fail on wBCE by construction:")
    print("  logit(0.5) = 0 is its fixed point, so it cannot shift a curve that")
    print("  is depressed on both sides of 0.5.")

    print("\n" + "=" * 78)
    print("Transfer cost -- fitted on seen and applied, vs fitted on unseen")
    print("=" * 78)
    for loss in MD.LOSS_ORDER:
        for cal in ("temperature", "platt"):
            a = df[(df.loss == loss) & (df.calibrator == cal)
                   & (df.fit_on == "seen") & (df.eval_on == "unseen")].ece.mean()
            b = df[(df.loss == loss) & (df.calibrator == cal)
                   & (df.fit_on == "unseen")].ece.mean()
            if np.isnan(a) or np.isnan(b):
                continue
            print(f"  {loss:<6s} {cal:<12s} transferred {a:.4f}   "
                  f"oracle {b:.4f}   cost {a - b:+.4f}")


def main():
    if not config.check():
        raise SystemExit("Fix the config paths first.")
    images = MD.available_images()
    if not images:
        raise SystemExit("Cache is empty -- run `python predict.py` first.")

    print(f"\nRecalibrating {len(MD.all_tags())} models")
    df = run(images)
    df.to_csv(os.path.join(config.RESULTS, "exp3_recalibration.csv"),
              index=False)
    summarise(df)

    print("\nFigure ...")
    fig_recalibration(images, df,
                      os.path.join(config.FIGURES, "fig3_recalibration.png"))
    print(f"\nTables -> {config.RESULTS}")


if __name__ == "__main__":
    main()
