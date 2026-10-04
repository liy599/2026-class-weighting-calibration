"""
Metrics for vegetation line detection and for probability calibration.

Detection metrics (precision / recall / F1 / IoU / FOM) reproduce what
Conor reports, so our numbers stay comparable to his.

Calibration metrics (ECE, MCE, reliability curves, Brier) are new -- they
are what the uncertainty paper is actually about.

A note on class imbalance that matters for the paper:
edge pixels are roughly 0.5% of an image, so a global ECE is dominated by
easy background pixels and looks deceptively good. We therefore also report
ECE restricted to a narrow band around the true line (`band_mask`), which is
the region anyone actually cares about.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


# --------------------------------------------------------------- DETECTION

def confusion(pred, target):
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    tp = np.sum(pred & target)
    tn = np.sum(~pred & ~target)
    fp = np.sum(pred & ~target)
    fn = np.sum(~pred & target)
    return int(tp), int(tn), int(fp), int(fn)


def detection_metrics(pred, target):
    """Binary prediction vs binary target."""
    tp, tn, fp, fn = confusion(pred, target)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def figure_of_merit(pred, target, alpha=1.0 / 9.0):
    """Pratt's Figure of Merit -- distance-tolerant edge score.

    Rewards predicted edge pixels for being *close* to a true edge pixel,
    rather than demanding exact overlap. This is the metric Conor headlines,
    and it is the right one for a line that is allowed to be a pixel off.
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)

    n_pred, n_true = int(pred.sum()), int(target.sum())
    if n_pred == 0 or n_true == 0:
        return 0.0

    dist = distance_transform_edt(~target)
    d = dist[pred]
    fom = np.sum(1.0 / (1.0 + alpha * d * d)) / max(n_pred, n_true)
    return float(fom)


def band_mask(target, radius=10):
    """Pixels within `radius` of the true line.

    Used to restrict calibration analysis to the neighbourhood of the line,
    where the model's confidence actually has consequences.
    """
    dist = distance_transform_edt(~np.asarray(target).astype(bool))
    return dist <= radius


# ------------------------------------------------------------- CALIBRATION

def reliability_curve(probs, labels, n_bins=15, strategy="uniform"):
    """Observed frequency vs predicted probability, per bin.

    Returns a dict with per-bin confidence, accuracy and count. Bins with
    no samples are dropped, so the arrays are ready to plot directly.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels).astype(np.float64).ravel()

    if strategy == "quantile":
        # Equal-count bins: better behaved when probabilities pile up near 0
        edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    idx = np.digitize(probs, edges[1:-1], right=False)

    conf, acc, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        conf.append(float(probs[m].mean()))
        acc.append(float(labels[m].mean()))
        cnt.append(n)

    return {
        "confidence": np.array(conf),
        "accuracy": np.array(acc),
        "count": np.array(cnt),
        "edges": edges,
    }


def expected_calibration_error(probs, labels, n_bins=15, strategy="uniform"):
    """ECE: count-weighted mean gap between confidence and observed frequency.

    0 is perfectly calibrated. Positive values mean the two disagree; use
    `signed_calibration_error` to find out in which direction.
    """
    r = reliability_curve(probs, labels, n_bins=n_bins, strategy=strategy)
    if len(r["count"]) == 0:
        return float("nan")
    w = r["count"] / r["count"].sum()
    return float(np.sum(w * np.abs(r["confidence"] - r["accuracy"])))


def maximum_calibration_error(probs, labels, n_bins=15, strategy="uniform"):
    """The worst single bin. Catches failures that ECE averages away."""
    r = reliability_curve(probs, labels, n_bins=n_bins, strategy=strategy)
    if len(r["count"]) == 0:
        return float("nan")
    return float(np.max(np.abs(r["confidence"] - r["accuracy"])))


def signed_calibration_error(probs, labels, n_bins=15, strategy="uniform"):
    """Mean of (confidence - accuracy), keeping the sign.

    Positive  -> the model is OVERCONFIDENT (claims more than it delivers).
    Negative  -> underconfident.

    This is the number the paper's central claim rests on: if the model is
    overconfident on unseen beaches but calibrated on seen ones, that is the
    finding.
    """
    r = reliability_curve(probs, labels, n_bins=n_bins, strategy=strategy)
    if len(r["count"]) == 0:
        return float("nan")
    w = r["count"] / r["count"].sum()
    return float(np.sum(w * (r["confidence"] - r["accuracy"])))


def brier_score(probs, labels):
    """Mean squared error of the probabilities. Lower is better."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels).astype(np.float64).ravel()
    return float(np.mean((probs - labels) ** 2))


# -------------------------------------------------------------- UNCERTAINTY

def predictive_entropy(mean_prob, eps=1e-12):
    """Total uncertainty of the ensemble mean, per pixel (binary entropy)."""
    p = np.clip(np.asarray(mean_prob, dtype=np.float64), eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def mutual_information(member_probs, eps=1e-12):
    """Epistemic (model) uncertainty: total entropy minus mean member entropy.

    High where the ensemble members *disagree* -- which is exactly the
    "the model does not know this beach" signal we are looking for.

    member_probs: array of shape (n_members, H, W)
    """
    member_probs = np.asarray(member_probs, dtype=np.float64)
    mean_p = member_probs.mean(axis=0)

    total = predictive_entropy(mean_p, eps=eps)

    p = np.clip(member_probs, eps, 1 - eps)
    per_member = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    aleatoric = per_member.mean(axis=0)

    return total - aleatoric, total, aleatoric


def member_std(member_probs):
    """Plain standard deviation across ensemble members, per pixel."""
    return np.asarray(member_probs, dtype=np.float64).std(axis=0)
