# Extra Weight on Rare Edge Pixels Makes Coastal Vegetation Line Models Overconfident

[![Paper](https://img.shields.io/badge/Paper-under%20review-lightgrey)](#-citation)
[![Dataset](https://img.shields.io/badge/Dataset-SIVE%20(Zenodo)-blue)](https://zenodo.org/records/17122999)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB)](requirements.txt)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Code, results and figures for a study of **probability calibration** in deep
learning models that detect the coastal vegetation line in Sentinel 2 images.

---

## 📖 Introduction

Deep edge detection models can find the vegetation line (the seaward edge of
dune plants) in satellite images, and the result is used to measure coastal
erosion. For every pixel the model gives a probability that the pixel lies on
the line. Whether these probabilities can be trusted had not been checked.

Edge pixels are very rare (about 0.6 percent of a training crop), and the
usual remedy is **class weighting**: a multiplier that makes mistakes on the
rare class cost more during training. We train **twelve models** with the
original author's code and data, and show that this weight makes the models
strongly overconfident, brings no gain in accuracy, and leaves no usable
default threshold.

The study builds on the [SIVE dataset](https://zenodo.org/records/17122999)
and the [original code](https://github.com/conorosully/sentinel2-vegetation-line)
by Conor O'Sullivan. We change one thing: which loss function the training
sweep is asked for.

## 🚀 Key findings

| Loss | Signed calibration error (seen) | ECE | Best FOM | Operating window | Best threshold |
|---|---:|---:|---:|---:|---|
| **BCE** (no weight) | −0.0013 | 0.003 | 0.764 | 19 % | 0.06 to 0.42 |
| **DICE** | +0.0142 | 0.037 | 0.731 | 68 % | 0.02 to 0.98 |
| **wBCE** (weight 166) | **+0.1505** | **0.151** | 0.737 | **8 %** | 0.96 to 0.98 |

Signed error is the mean over the four backbone and guidance settings;
positive means overconfident. The operating window is the share of decision
thresholds for which the figure of merit (FOM) stays within 0.02 of its best.

- **The weight is the cause, not the loss family.** Plain, unweighted binary
  cross entropy (BCE) is better calibrated than Dice loss. Adding the weight
  makes it the worst. We trained BCE ourselves, because the original sweep
  only ran weighted BCE and Dice, which differ in two ways at once.
- **The loss itself predicts the effect.** Weighted cross entropy with weight
  `w` is minimised at `q = w p / (1 − p + w p)`. This needs no fitted
  parameters and explains about nine tenths of the measured error.
- **The weight provides no benefit.** At its own best threshold, unweighted BCE
  is the most accurate loss, and the best threshold of the weighted model
  (0.96 to 0.98) does not include the default of 0.5.
- **The effect is not an artefact of how it is measured.** It holds in all 48
  combinations of bin count, binning rule and distance from the line.
- **Repair after training:** Platt scaling lowers the ECE of the weighted
  models on the unseen beach from 0.120 to 0.016. Temperature scaling cannot
  work, because it cannot move the probability 0.5, and it raises the ECE to
  0.140.
- **Best model:** trainable, guided, DICE. **Worst:** frozen, unguided, wBCE.

## 🧪 Experimental design

Twelve models, a full 2 × 2 × 3 factorial. All use the same HED network on a
ResNet50 backbone pretrained on BigEarthNet.

| Factor | Levels |
|---|---|
| Backbone | `frozen`, `trainable` |
| Guidance band (a hand drawn rough map of the coast, as a fifth input) | `guided`, `unguided` |
| Loss | `BCE` (our addition), `wBCE` (weight 166), `DICE` |

The original eight models (`wBCE` and `DICE`) come from the original training
script. The four `BCE` models are trained by `src/train_bce_control.py` with
**every other setting identical** (the same 70/30 training and validation
split, fixed by seed 42, the same learning rate search and the same early
stopping). The seed fixes only the data split. The initial weights of the new
layers and the order of the training batches are not seeded, so each model is
a single training run and the variation between runs is not measured. Calibration is measured within 10 pixels of the
true line, because a whole image score is dominated by easy background.

## 🗂 Repository structure

```
.
├── src/
│   ├── config.py                  all paths (portable, see Setup)
│   ├── make_points_dict.py        builds points_dict.npy, which SIVE does not ship
│   ├── train_bce_control.py       trains the four missing unweighted BCE models
│   ├── predict.py                 caches a probability map for every model and scene
│   ├── metrics.py                 detection and calibration metrics
│   ├── models.py                  model inventory and loss taxonomy
│   ├── exp1_loss_calibration.py   the separation and its cause
│   ├── exp2_threshold.py          effect of the weight on accuracy and threshold: operating window
│   ├── exp3_recalibration.py      repair after training: temperature vs Platt scaling
│   ├── exp4_sensitivity.py        robustness over 48 measurement settings
│   ├── exp5_model_ranking.py      multi criteria comparison of the twelve models
│   ├── figures_paper.py           the four figures used in the paper
│   └── paper_numbers.py           every number quoted in the paper, checked against the manuscript
├── notebooks/
│   ├── model_comparison.ipynb     all tables and figures, run live
│   └── model_comparison.py        its source (jupytext format)
├── results/                       csv tables written by the experiments
├── figures/                       paper_fig*.pdf / .png, plus per experiment plots
├── patches/                       the two line change to the original repository
├── requirements.txt
└── LICENSE
```

`figures/paper_fig*.pdf` are the figures used in the paper. The other
`figures/fig*.png` are diagnostic plots written by the experiment scripts.

## ⚙️ Setup

Not included in this repository, because of size or licence: the SIVE data,
the twelve trained weight files (about 94 MB each), and the probability cache
(about 580 MB). All of them can be rebuilt.

**1. Environment.** Python 3.10.

```bash
pip install -r requirements.txt
```

**2. This repository, and the original one next to it.**

```bash
git clone https://github.com/liy599/2026-class-weighting-calibration.git
```

```bash
git clone https://github.com/conorosully/sentinel2-vegetation-line.git
```

```
some_folder/
├── sentinel2-vegetation-line/          the original repository
└── 2026-class-weighting-calibration/   this repository
```

If it is somewhere else, set `SIVE_REPO` instead (PowerShell:
`$env:SIVE_REPO = "D:\work\sentinel2-vegetation-line"`).

**3. The SIVE data.** Download it from
[Zenodo](https://zenodo.org/records/17122999) (CC BY 4.0) and unzip so that
`sentinel2-vegetation-line/data/SIVE/` contains `training/`, `test_1/` (seen
locations) and `test_2/` (the unseen beach).

**4. One change to the original code.** It was written on a Mac and selects the
`mps` device in two places. The patch switches these to `cuda`:

```bash
cd sentinel2-vegetation-line
git apply ../coastal-uncertainty/patches/sentinel2-vegetation-line_cuda.patch
```

On a Mac, or without a GPU, edit those two lines to `mps` or `cpu` instead.
An RTX 50 series card needs a PyTorch nightly build.

**5. `points_dict.npy`.** The original code needs it, but it is not shipped
with SIVE. It is a regular grid of crop centres (spacing 72 pixels):

```bash
python src/make_points_dict.py
```

**6. Check everything.**

```bash
python src/config.py
```

## ▶️ Reproducing the results

Everything after step 3 below needs **no GPU** and runs in minutes, once the
probability cache exists.

**1. Train the twelve models** (about 11 hours on one laptop GPU in total).
From `sentinel2-vegetation-line/src`, this trains the original eight:

```bash
python train.py --model_name SIVE_04JUN2025 --model_type HED --backbone_dataset BigEarthNet --batch_size 16 --epochs 100 --train_path ../data/SIVE/training/ --save_path ../models/SIVE_04JUN2025/ --device cuda --early_stopping 10
```

Create `models/SIVE_04JUN2025/` first. Then, from this repository, the four
unweighted models (about one hour each; models that already exist are skipped):

```bash
python src/train_bce_control.py
```

**2. Cache the probability maps** (300 files, about 580 MB, 1 to 2 hours):

```bash
python src/predict.py
```

**3. Run the experiments and build the figures:**

```bash
python src/exp1_loss_calibration.py
python src/exp2_threshold.py
python src/exp3_recalibration.py
python src/exp4_sensitivity.py
python src/exp5_model_ranking.py
python src/figures_paper.py
```

```bash
python src/paper_numbers.py
```

`exp5` reads the output of `exp1` and `exp2`, so run them in this order.
`paper_numbers.py` recomputes every number quoted in the paper, prints where
each one comes from, and checks it against the value in the manuscript
(58 numbers; all match). It also writes the rows of the paper's two tables.

Given the same cache, the result tables are reproduced exactly. Training is
not deterministic: the seed (42) fixes only the split into training and
validation data, and not the initial weights or the batch order, so retraining
gives similar but not identical models.

**Or just look:** `results/` holds every table, and
`notebooks/model_comparison.ipynb` shows all of them and the four figures
with the reasoning. To rebuild the notebook after editing its source:

```bash
jupytext --to notebook notebooks/model_comparison.py
```

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/model_comparison.ipynb
```

## 📊 Output files

| File | Content |
|---|---|
| `results/exp1_per_model.csv` | detection and calibration for every model and test scene |
| `results/exp1_table1.csv` | the twelve model table (FOM and signed error, both test sets) |
| `results/exp1_one_way.csv` | ratio of signed error to ECE (1 means a one directional bias) |
| `results/exp2_threshold.csv` | FOM at the default and best threshold, operating window |
| `results/exp3_recalibration.csv` | temperature and Platt scaling, fitted on seen, tested on unseen |
| `results/exp4_sensitivity.csv`, `exp4_separation.csv` | the 48 measurement settings |
| `results/exp5_ranking.csv` | five criteria and the composite score for every model |
| `results/paper_numbers.txt` | every number quoted in the paper, with its source and a check |
| `results/paper_table1_rows.tex`, `paper_table2_rows.tex` | the rows of the paper's two tables |

## ⚠️ Limitations

- The best thresholds were chosen on the data they are scored on. They bound
  what tuning could reach, so the recommendation rests on the operating window.
- The unseen test set is a single beach (Rossnowlagh, 16 scenes).
- One class weight (166) was tested, and each model was trained once.
- One network design on one dataset.

## 📚 Citation

If you use this code or these results, please cite our paper (under review,
this entry will be updated):

```bibtex
@misc{li2026extraweight,
  title  = {Extra Weight on Rare Edge Pixels Makes Coastal Vegetation Line Models Overconfident},
  author = {Li, Yukun and Dey, Prasanjit and Pakrashi, Arjun and Dev, Soumyabrata},
  year   = {2026},
  note   = {Manuscript under review}
}
```

and the original work and dataset that this study builds on:

```bibtex
@article{osullivan2026subpixel,
  title   = {Detecting Subpixel Changes in the Coastal Vegetation Line With Sentinel-2 Imagery},
  author  = {O'Sullivan, Conor and Monteys, Xavier and Dev, Soumyabrata},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume  = {19},
  pages   = {10421--10437},
  year    = {2026},
  doi     = {10.1109/JSTARS.2026.3661632}
}

@misc{osullivan2025sive,
  title     = {The Sentinel-2 Irish Vegetation Edge (SIVE) Dataset},
  author    = {O'Sullivan, Conor},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17122999}
}
```

## 🙏 Acknowledgements

This work uses the dataset and code of
[Conor O'Sullivan](https://github.com/conorosully/sentinel2-vegetation-line)
(code under the MIT licence, data under CC BY 4.0). It was conducted with the
financial support of Research Ireland under Grant No. 13/RC/2106_P2 at the
ADAPT Research Centre.

## 📄 License

Code: MIT, see [LICENSE](LICENSE). The SIVE data keeps its own licence
(CC BY 4.0).
