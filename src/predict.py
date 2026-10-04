"""
Cache probability maps for every (model, image) pair.

This is the expensive step of the whole project: 8 models x 25 images.
We run it once, save every probability map as .npy, and every later
experiment reads from the cache instead of re-running the network.

Run this as a script:
    python predict.py

It is safe to interrupt and restart -- anything already cached is skipped.
"""

import os
import glob
import time
import numpy as np

import config


def model_tag(model_path):
    """'SIVE_04JUN2025_HED_BigEarthNet_trainable_guided_wBCE.pth'
       -> 'trainable_guided_wBCE'   (the part that actually varies)"""
    base = os.path.basename(model_path).replace(".pth", "")
    parts = base.split("_")
    # SIVE_<date>_HED_<backbone>_<freeze>_<guidance>_<loss>
    return "_".join(parts[4:])


def image_tag(image_path):
    """'.../20171129T113419_bull_island.npy' -> '20171129T113419_bull_island'"""
    return os.path.basename(image_path).replace(".npy", "")


def cache_path(m_tag, i_tag):
    return os.path.join(config.CACHE, f"prob__{m_tag}__{i_tag}.npy")


def list_test_images():
    """Return [(path, split)] for every test image, split in {'seen','unseen'}."""
    out = []
    for p in sorted(glob.glob(os.path.join(config.TEST_SEEN, "*.npy"))):
        out.append((p, "seen"))
    for p in sorted(glob.glob(os.path.join(config.TEST_UNSEEN, "*.npy"))):
        out.append((p, "unseen"))
    return out


def list_models():
    return sorted(glob.glob(os.path.join(config.MODELS, "*.pth")))


def build_cache(overwrite=False):
    """Run every model over every test image, saving probability maps.

    We ask for probabilistic output (threshold=None) and no skeletonisation
    (post_process=False), because uncertainty analysis needs the raw
    probabilities -- thresholding throws away exactly the information we want.
    """
    old_cwd = config.enter_sive()
    try:
        import utils
        import evaluation as ev

        points_dict = np.load(config.POINTS_DICT, allow_pickle=True).item()

        models = list_models()
        images = list_test_images()
        print(f"{len(models)} models x {len(images)} images "
              f"= {len(models) * len(images)} probability maps\n")

        for mi, model_path in enumerate(models, 1):
            m_tag = model_tag(model_path)

            # Skip the whole model if every one of its maps is already cached
            todo = [
                (p, s) for p, s in images
                if overwrite or not os.path.exists(cache_path(m_tag, image_tag(p)))
            ]
            if not todo:
                print(f"[{mi}/{len(models)}] {m_tag}: already cached, skipping")
                continue

            print(f"[{mi}/{len(models)}] {m_tag}: {len(todo)} maps to compute")
            model, meta = utils.get_model(model_path)

            for p, split in todo:
                i_tag = image_tag(p)
                t0 = time.time()
                try:
                    image, prob = ev.get_combined_pred(
                        model, meta, points_dict, p,
                        batch_size=1,
                        post_process=False,   # keep the raw map
                        threshold=None,       # probabilities, not 0/1
                    )
                except Exception as e:
                    print(f"      FAILED {i_tag}: {type(e).__name__}: {e}")
                    continue

                prob = np.asarray(prob, dtype=np.float32)
                np.save(cache_path(m_tag, i_tag), prob)

                # Ground truth is the last channel; save once per image
                tgt = os.path.join(config.CACHE, f"target__{i_tag}.npy")
                if not os.path.exists(tgt):
                    np.save(tgt, np.asarray(image[-1], dtype=np.uint8))

                print(f"      {i_tag:<42s} {split:<7s} "
                      f"{prob.shape}  {time.time() - t0:5.1f}s")

            del model

    finally:
        os.chdir(old_cwd)

    print("\nCache complete.")


def load_prob(m_tag, i_tag):
    return np.load(cache_path(m_tag, i_tag))


def load_target(i_tag):
    return np.load(os.path.join(config.CACHE, f"target__{i_tag}.npy"))


def cache_summary():
    """What is currently in the cache."""
    maps = glob.glob(os.path.join(config.CACHE, "prob__*.npy"))
    tgts = glob.glob(os.path.join(config.CACHE, "target__*.npy"))
    print(f"cached probability maps : {len(maps)}")
    print(f"cached targets          : {len(tgts)}")
    return len(maps), len(tgts)


if __name__ == "__main__":
    if not config.check():
        raise SystemExit("Fix the config paths first.")
    print()
    build_cache()
    print()
    cache_summary()
