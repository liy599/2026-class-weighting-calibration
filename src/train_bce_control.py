"""
Train the missing control: plain, UNWEIGHTED binary cross-entropy.

Why this exists
---------------
Experiment 1 found that every wBCE model is badly overconfident and every
DICE model is nearly calibrated, and attributed this to `pos_weight = 166`.
That attribution is not yet earned, because wBCE differs from DICE in *two*
ways at once: a different loss family, and the 166x positive-class weight.
Nothing in the existing eight models separates the two.

Plain BCE is the missing cell of that design. It shares the loss family with
wBCE and the absence of weighting with DICE, so its calibration says which
of the two is responsible:

    plain BCE calibrated    -> the weighting is the cause
    plain BCE miscalibrated -> the loss family is the cause, and the
                               paper's stated mechanism is wrong

This also settles an apparent conflict with the segmentation literature,
which reports the opposite ordering (DICE overconfident, cross-entropy well
calibrated) -- because that literature compares DICE against *unweighted*
cross-entropy, which is exactly the model trained here.

`train.py` already implements this loss (`args.loss == "BCE"`), but its
sweep loop only ever iterates `["wBCE", "DICE"]`, so these weights were
never produced. We do not edit Conor's repo; we import from it and run the
one arm it skipped, with every other setting identical to the original run
-- same seed, same split, same learning-rate sweep -- so the new models are
directly comparable to the existing eight.

Run:
    python train_bce_control.py
"""

import argparse
import os
import time

import config


# All four configurations, so the 2x3 design is complete.
#
# The first pass trained only the guided pair, on the grounds that the
# unguided models were too weak for a calibration comparison to mean much.
# That reasoning was wrong: detection quality and calibration are independent
# axes. frozen_unguided_DICE reaches only FOM 0.57 yet has one of the best
# calibration errors in the whole set (+0.013), while frozen_unguided_wBCE is
# the single most overconfident model measured (+0.205). The unguided arm is
# where the effect is *largest*, and it is the harder task, so confirming the
# mechanism there is a stronger test rather than a weaker one.
#
# Anything already on disk is skipped, so re-running this is safe.
CONFIGS = [
    dict(freeze_backbone=False, guidance=True),    # trainable_guided_BCE
    dict(freeze_backbone=True,  guidance=True),    # frozen_guided_BCE
    dict(freeze_backbone=False, guidance=False),   # trainable_unguided_BCE
    dict(freeze_backbone=True,  guidance=False),   # frozen_unguided_BCE
]


def build_args():
    """Exactly the arguments of the original 8-model run, but loss = BCE.

    Reproduced from the command used for the original eight models (see the
    README, section "Reproducing the results"). seed and split in
    particular must match, or the new models see a different train/valid
    partition and the comparison is contaminated.
    """
    return argparse.Namespace(
        model_name="SIVE_04JUN2025",
        sample=False,
        model_type="HED",
        backbone_dataset="BigEarthNet",
        freeze_backbone=False,          # set per config below
        guidance=True,                  # set per config below
        batch_size=16,
        epochs=100,
        split=0.7,
        early_stopping=10,
        train_path=os.path.join(config.DATA, "training") + os.sep,
        save_path=config.MODELS,
        device="cuda",
        seed=config.SEED,
        loss="BCE",
    )


def main():
    if not config.check():
        raise SystemExit("Fix the config paths first.")

    old_cwd = config.enter_sive()
    try:
        import torch
        import train as T

        args = build_args()
        args.device = torch.device(args.device)

        print(f"\nDevice: {args.device}")
        if args.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Saving to: {args.save_path}")
        print(f"Loss: {args.loss} (unweighted -- this is the control)\n")

        for i, cfg in enumerate(CONFIGS, 1):
            args.freeze_backbone = cfg["freeze_backbone"]
            args.guidance = cfg["guidance"]

            s1 = "frozen" if args.freeze_backbone else "trainable"
            s2 = "guided" if args.guidance else "unguided"
            tag = f"{s1}_{s2}_{args.loss}"

            target = os.path.join(
                args.save_path,
                f"{args.model_name}_{args.model_type}_"
                f"{args.backbone_dataset}_{tag}.pth")
            if os.path.exists(target):
                print(f"[{i}/{len(CONFIGS)}] {tag}: already trained, skipping")
                continue

            print("=" * 70)
            print(f"[{i}/{len(CONFIGS)}] {tag}")
            print("=" * 70)

            t0 = time.time()
            # load_data depends on guidance (it controls the channel count),
            # so it is reloaded per configuration rather than hoisted out.
            train_loader, valid_loader = T.load_data(args)
            T.train_model(train_loader, valid_loader, args)
            print(f"\n{tag} finished in {(time.time() - t0) / 60:.1f} min")

    finally:
        os.chdir(old_cwd)

    print("\nControl models trained. Next: `python predict.py` to cache their "
          "probability maps, then `python exp1_loss_calibration.py`.")


if __name__ == "__main__":
    main()
