"""
The model inventory, shared by every experiment.

This study analyses individual models trained with the original author's
code and data. It deliberately does not build ensembles: every claim it
makes is a per-model claim, and averaging probability maps would blur the
loss-function effect that is the whole point.

The design under test is a 2x2x3 factorial: two backbone treatments (frozen,
trainable) crossed with two guidance settings (guided, unguided) and three
losses (BCE, DICE, wBCE). wBCE differs from BCE only by `pos_weight = 166`,
which is what lets the weighting be separated from the loss family.
"""

import os
import glob
import numpy as np

import config
import predict as P


#: Loss -> (family, weighted). The paper's whole argument is that the
#: second column, not the first, is what governs calibration.
LOSS_FAMILY = {"BCE": "cross-entropy", "wBCE": "cross-entropy", "DICE": "dice"}
LOSS_WEIGHTED = {"BCE": False, "wBCE": True, "DICE": False}


#: Plot order and styling, used by every figure so the paper stays coherent.
LOSS_ORDER = ["BCE", "DICE", "wBCE"]
LOSS_STYLE = {
    "BCE":  ("#1a9850", "D", "BCE (unweighted)"),
    "DICE": ("#2166ac", "o", "DICE"),
    "wBCE": ("#b2182b", "s", "wBCE (weighted $\\times$166)"),
}


# ------------------------------------------------------------------ TAGS

def all_tags():
    """Every trained model, as 'trainable_guided_DICE'-style tags."""
    return sorted(
        P.model_tag(p) for p in glob.glob(os.path.join(config.MODELS, "*.pth"))
    )


def loss_of(tag):
    """'trainable_guided_wBCE' -> 'wBCE'."""
    return tag.rsplit("_", 1)[-1]


def config_of(tag):
    """'trainable_guided_wBCE' -> 'trainable_guided'."""
    return tag.rsplit("_", 1)[0]


def is_guided(tag):
    """Careful: 'unguided' contains 'guided' as a substring, so the naive
    test silently returns every model."""
    return "unguided" not in tag


def tags_with(loss=None, guided=None, freeze=None):
    tags = all_tags()
    if loss is not None:
        tags = [t for t in tags if loss_of(t) == loss]
    if guided is not None:
        tags = [t for t in tags if is_guided(t) == guided]
    if freeze is not None:
        tags = [t for t in tags if t.startswith(freeze)]
    return sorted(tags)


def complete_configs():
    """Configurations for which all three losses were trained.

    Only these support the controlled comparison the paper rests on -- the
    unguided arm was never run for unweighted BCE, so including it would
    compare losses across different configurations.
    """
    by_cfg = {}
    for t in all_tags():
        by_cfg.setdefault(config_of(t), set()).add(loss_of(t))
    return sorted(c for c, ls in by_cfg.items() if set(LOSS_ORDER) <= ls)


# ---------------------------------------------------------------- IMAGES

def available_images():
    """[(image_tag, split)] for every scene with a cached target."""
    split_of = {P.image_tag(p): s for p, s in P.list_test_images()}
    out = []
    for f in sorted(glob.glob(os.path.join(config.CACHE, "target__*.npy"))):
        i_tag = os.path.basename(f)[len("target__"):-len(".npy")]
        out.append((i_tag, split_of.get(i_tag, "unknown")))
    return out


def site_of(i_tag):
    """'20171129T113419_bull_island' -> 'bull_island'."""
    return i_tag.split("_", 1)[1] if "_" in i_tag else i_tag


# --------------------------------------------------------------- POOLING

def pooled_pixels(tags, images, split, band_fn, subsample=None, seed=0):
    """Concatenate (probability, label) over models and scenes.

    Pools pixels from each model *separately* rather than averaging their
    probability maps. Averaging would be an ensemble, and an ensemble of
    differently-calibrated members has calibration properties of its own
    that say nothing about any single model. Pooling answers the question
    the paper actually asks: what does a model trained with this loss look
    like?
    """
    ps, ys = [], []
    rng = np.random.default_rng(seed)
    for tag in tags:
        for i_tag, s in images:
            if s != split:
                continue
            path = P.cache_path(tag, i_tag)
            if not os.path.exists(path):
                continue
            prob = P.load_prob(tag, i_tag)
            target = P.load_target(i_tag).astype(bool)
            m = band_fn(target)
            p, y = prob[m], target[m].astype(np.float64)
            if subsample is not None and len(p) > subsample:
                idx = rng.choice(len(p), subsample, replace=False)
                p, y = p[idx], y[idx]
            ps.append(p)
            ys.append(y)
    if not ps:
        return np.array([]), np.array([])
    return np.concatenate(ps), np.concatenate(ys)


def describe():
    """Print the design matrix -- run this to see what exists."""
    tags = all_tags()
    cfgs = sorted({config_of(t) for t in tags})
    print(f"{len(tags)} models over {len(cfgs)} configurations\n")
    header = f"{'configuration':<22s}" + "".join(f"{l:>8s}" for l in LOSS_ORDER)
    print(header)
    print("-" * len(header))
    for c in cfgs:
        have = {loss_of(t) for t in tags if config_of(t) == c}
        row = f"{c:<22s}" + "".join(
            f"{'yes' if l in have else '--':>8s}" for l in LOSS_ORDER)
        print(row)
    print(f"\ncomplete configurations (all three losses): "
          f"{complete_configs()}")


if __name__ == "__main__":
    describe()
