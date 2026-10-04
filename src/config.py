"""
Central configuration for the class weighting and calibration study.

Every path used by the experiments lives here. Nothing is hard coded to one
machine: the project folder is found from this file's own location, and the
folder holding the original author's repository (code, data and model
weights) is found from, in this order,

  1. the environment variable SIVE_REPO, if it is set;
  2. a folder called `sentinel2-vegetation-line` that sits next to this
     project folder.

So a layout like

    some_folder/
        sentinel2-vegetation-line/     <- the original repository
        coastal-uncertainty/           <- this repository

works with no configuration at all. Anywhere else, set SIVE_REPO, for
example (Windows PowerShell):

    $env:SIVE_REPO = "D:\\work\\sentinel2-vegetation-line"
"""

import glob
import os
import sys

# ---------------------------------------------------------------- ROOT PATHS

# This project: the folder that contains src/, results/, figures/ ...
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The original author's repository. We import his code from here and we only
# ever read his data; the one change we make to his code is documented in
# patches/ and in the README.
SIVE_REPO = os.environ.get(
    "SIVE_REPO",
    os.path.join(os.path.dirname(PROJECT), "sentinel2-vegetation-line"),
)
SIVE_SRC = os.path.join(SIVE_REPO, "src")

# ---------------------------------------------------------------- DATA PATHS

DATA = os.path.join(SIVE_REPO, "data", "SIVE")
TEST_SEEN = os.path.join(DATA, "test_1")      # beaches also used in training
TEST_UNSEEN = os.path.join(DATA, "test_2")    # Rossnowlagh, never seen
POINTS_DICT = os.path.join(DATA, "points_dict.npy")   # see make_points_dict.py

# Trained model weights (twelve variants).
MODELS = os.path.join(SIVE_REPO, "models", "SIVE_04JUN2025")

# ------------------------------------------------------------- OUTPUT PATHS

CACHE = os.path.join(PROJECT, "cache")        # cached probability maps (large)
RESULTS = os.path.join(PROJECT, "results")    # csv tables
FIGURES = os.path.join(PROJECT, "figures")    # figures

for _d in (CACHE, RESULTS, FIGURES):
    os.makedirs(_d, exist_ok=True)

# ------------------------------------------------------------------ RUNTIME

DEVICE = "cuda"
CROP_SIZE = 144
SEED = 42


# ------------------------------------------------- MAKE THE ORIGINAL CODE IMPORTABLE

def add_sive_to_path():
    """Put the original repository's src/ on sys.path so `import utils`,
    `import evaluation` work."""
    if SIVE_SRC not in sys.path:
        sys.path.insert(0, SIVE_SRC)


def enter_sive():
    """chdir into the original repository's src/ and return where we were.

    His evaluation code resolves data paths relative to the working
    directory, so we have to run from there. Always pair with
    os.chdir(old) afterwards.
    """
    add_sive_to_path()
    old = os.getcwd()
    os.chdir(SIVE_SRC)
    return old


# ------------------------------------------------------------------ SANITY

def check():
    """Verify every path exists. Run this first: it saves hours."""
    problems = []
    for name, path in [
        ("original repo", SIVE_REPO),
        ("original src", SIVE_SRC),
        ("test_1 (seen)", TEST_SEEN),
        ("test_2 (unseen)", TEST_UNSEEN),
        ("model directory", MODELS),
        ("points_dict.npy", POINTS_DICT),
    ]:
        if not os.path.exists(path):
            problems.append(f"  MISSING  {name}: {path}")

    if problems:
        print("Config problems found:")
        print("\n".join(problems))
        print("\nSee the README, section 'Setup'. If the original repository is "
              "somewhere else, set the SIVE_REPO environment variable.")
        return False

    n_models = len(glob.glob(os.path.join(MODELS, "*.pth")))
    n_seen = len(glob.glob(os.path.join(TEST_SEEN, "*.npy")))
    n_unseen = len(glob.glob(os.path.join(TEST_UNSEEN, "*.npy")))

    print("Config OK")
    print(f"  original repo : {SIVE_REPO}")
    print(f"  models        : {n_models}   (expected 12)")
    print(f"  seen          : {n_seen} images   (expected 9)")
    print(f"  unseen        : {n_unseen} images  (expected 16)")
    print(f"  cache         : {CACHE}")
    print(f"  results       : {RESULTS}")
    return True


if __name__ == "__main__":
    check()
