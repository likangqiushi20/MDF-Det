"""Train SPGF V4 on six-AOI-excluded native-resolution shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from .model import (GroupNormalization, access_mae, build_spgf_v4,
                        motion_mae, semantic_prior_v2_loss)
    from .training_utils import ShardSequence, completed_epochs
except ImportError:  # Allow: python SPGF/train.py ...
    from model import (GroupNormalization, access_mae, build_spgf_v4,
                       motion_mae, semantic_prior_v2_loss)
    from training_utils import ShardSequence, completed_epochs


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-filters", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weak-negative-weight", type=float, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--initial-epoch", type=int, default=-1)
    parser.add_argument("--cache-shards", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.data / "dataset_summary.json").read_text(encoding="utf-8"))
    if summary.get("version") != "SPGF-V4" or not summary.get("all_shards_audit_passed"):
        raise RuntimeError("Training requires an audited SPGF-V4 dataset")
    if set(summary.get("held_out_aois", [])) != {"01", "02", "03", "34", "40", "41"}:
        raise RuntimeError("V4 dataset does not exclude all six publication AOIs")
    weak_weight = (float(args.weak_negative_weight) if args.weak_negative_weight is not None
                   else float(summary["configuration"]["weak_negative_weight"]))
    train = ShardSequence(args.data, "train", args.batch_size, True, weak_weight,
                          20260902, args.cache_shards)
    val = ShardSequence(args.data, "val", args.batch_size, False, weak_weight,
                        20260903, args.cache_shards)
    custom = {"GroupNormalization": GroupNormalization,
              "semantic_prior_v2_loss": semantic_prior_v2_loss,
              "access_mae": access_mae, "motion_mae": motion_mae}
    model = (tf.keras.models.load_model(args.resume, custom_objects=custom)
             if args.resume else build_spgf_v4(args.base_filters))
    model.compile(tf.keras.optimizers.Adam(args.lr), loss=semantic_prior_v2_loss,
                  metrics=[access_mae, motion_mae])
    history_path = args.output / "history.csv"
    initial_epoch = (completed_epochs(history_path) if args.initial_epoch < 0 and args.resume
                     else max(args.initial_epoch, 0))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(args.output / "best_semantic_prior_v4.h5"),
                                           monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(args.output / "latest_semantic_prior_v4.h5"),
                                           save_best_only=False, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                                             min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(str(history_path), append=bool(args.resume)),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(train, validation_data=val, epochs=args.epochs,
                        initial_epoch=initial_epoch, callbacks=callbacks,
                        workers=1, use_multiprocessing=False)
    model.save(args.output / "final_restored_best_semantic_prior_v4.h5")
    losses = history.history.get("val_loss", [])
    report = {
        "version": "SPGF-V4", "held_out_aois": summary["held_out_aois"],
        "dataset_audit_passed": summary["all_shards_audit_passed"],
        "initial_epoch": initial_epoch, "completed_this_run": len(losses),
        "best_epoch_this_run": int(initial_epoch + np.argmin(losses) + 1) if losses else None,
        "best_val_loss_this_run": float(np.min(losses)) if losses else None,
        "train_samples": train.sample_count, "val_samples": val.sample_count,
        "train_batches": len(train), "val_batches": len(val),
        "batch_size": args.batch_size, "weak_negative_weight": weak_weight,
    }
    (args.output / "training_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
