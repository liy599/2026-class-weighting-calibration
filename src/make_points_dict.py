"""
Create `points_dict.npy`, which the original author's code needs but which is
not shipped with the SIVE dataset.

What it is
----------
The original `get_combined_pred` cuts each test scene into overlapping
144 x 144 crops and runs the network on every crop. It decides where to cut
from a dictionary of (x, y) centre points:

    points_dict[image_id] = {'coastline': [(x, y), (x, y), ...]}

The key names the scene, and the value is a dictionary holding one list of
points. We use a regular grid with a spacing of 72 pixels, which is half a
crop, so neighbouring crops overlap by half and the whole scene is covered:

    x = 72, 144, 216, ... up to W - 72
    y = 72, 144, 216, ... up to H - 72      (points listed row by row)

The grid does not depend on where the coast is. Using the true line to place
crops would leak the answer into the test procedure.

How we know this is right
-------------------------
The file used for the study was first made by hand. This script reproduces
that rule, and was checked to give a dictionary identical to it for all 25
test scenes (9 seen, 16 unseen). If a points_dict.npy already exists, this
script compares against it and says so, instead of overwriting.

Run
---
    python make_points_dict.py
"""

import glob
import os

import numpy as np

import config

STEP = 72          # half of the 144 pixel crop


def grid_points(height, width, step=STEP):
    """Centre points of the overlapping crops, row by row."""
    xs = range(step, width - step + 1, step)
    ys = range(step, height - step + 1, step)
    return [(x, y) for y in ys for x in xs]


def build():
    paths = sorted(glob.glob(os.path.join(config.TEST_SEEN, "*.npy"))
                   + glob.glob(os.path.join(config.TEST_UNSEEN, "*.npy")))
    if not paths:
        raise SystemExit(f"No test images found in\n  {config.TEST_SEEN}\n  "
                         f"{config.TEST_UNSEEN}\nDownload SIVE first (README, Setup).")
    out = {}
    for p in paths:
        shape = np.load(p, mmap_mode="r").shape      # (channels, H, W)
        key = os.path.basename(p)[: -len(".npy")]
        out[key] = {"coastline": grid_points(shape[-2], shape[-1])}
    return out


def main():
    new = build()
    print(f"{len(new)} scenes, "
          f"{sum(len(v['coastline']) for v in new.values())} points in total")

    if os.path.exists(config.POINTS_DICT):
        old = np.load(config.POINTS_DICT, allow_pickle=True).item()
        same = (set(old) == set(new)
                and all([tuple(t) for t in old[k]["coastline"]] == new[k]["coastline"]
                        for k in new))
        print(f"{config.POINTS_DICT} already exists and is "
              f"{'IDENTICAL to' if same else 'DIFFERENT from'} the generated grid.")
        if not same:
            print("Not overwriting. Delete it and run again to replace it.")
        return

    np.save(config.POINTS_DICT, new)
    print("wrote", config.POINTS_DICT)


if __name__ == "__main__":
    main()
