"""Train semantic-prior v2 from restartable full-frame NPZ shards."""
from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

try:
    from .model import (GroupNormalization, access_mae, build_semantic_prior_v2,
                        motion_mae, semantic_prior_v2_loss)
except ImportError:  # Allow: python SPGF/training_utils.py ...
    from model import (GroupNormalization, access_mae, build_semantic_prior_v2,
                       motion_mae, semantic_prior_v2_loss)


class ShardSequence(tf.keras.utils.Sequence):
    def __init__(self, root: Path, split: str, batch_size: int, augment: bool,
                 weak_negative_weight: float, seed: int, cache_size: int = 2):
        self.paths = sorted((root / split).glob("anchor_*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No shards found in {root / split}")
        self.batch_size = int(batch_size)
        self.augment = bool(augment)
        self.weak_negative_weight = float(weak_negative_weight)
        self.rng = np.random.default_rng(seed)
        self.cache_size = max(1, int(cache_size))
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.counts = []
        for path in self.paths:
            with np.load(path) as data:
                self.counts.append(int(len(data["images"])))
        self.sample_count = int(sum(self.counts))
        self.plan: list[tuple[int, np.ndarray]] = []
        self.on_epoch_end()

    def __len__(self):
        return len(self.plan)

    def _load(self, shard: int):
        if shard in self.cache:
            value = self.cache.pop(shard)
            self.cache[shard] = value
            return value
        with np.load(self.paths[shard]) as data:
            value = data["images"].copy(), data["targets"].copy()
        self.cache[shard] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def on_epoch_end(self):
        shard_order = np.arange(len(self.paths))
        if self.augment:
            self.rng.shuffle(shard_order)
        plan = []
        for shard in shard_order:
            order = np.arange(self.counts[shard])
            if self.augment:
                self.rng.shuffle(order)
            batches = [order[start:start + self.batch_size]
                       for start in range(0, len(order), self.batch_size)]
            if self.augment:
                self.rng.shuffle(batches)
            plan.extend((int(shard), batch) for batch in batches)
        self.plan = plan

    def __getitem__(self, index):
        shard, ids = self.plan[index]
        images, targets_u8 = self._load(shard)
        x = images[ids].astype(np.float32) / 255.0
        targets = targets_u8[ids].astype(np.float32) / 255.0
        if self.augment:
            for i in range(len(x)):
                turns = int(self.rng.integers(0, 4))
                x[i] = np.rot90(x[i], turns)
                targets[i] = np.rot90(targets[i], turns)
                if self.rng.random() < 0.5:
                    x[i], targets[i] = x[i, :, ::-1], targets[i, :, ::-1]
                if self.rng.random() < 0.5:
                    x[i], targets[i] = x[i, ::-1], targets[i, ::-1]
                x[i] = np.clip(x[i] * self.rng.uniform(0.85, 1.15) +
                               self.rng.uniform(-0.05, 0.05), 0, 1)
                if self.rng.random() < 0.20:
                    x[i] = cv2.GaussianBlur(x[i], (0, 0), float(self.rng.uniform(0.2, 1.0)))
                if self.rng.random() < 0.20:
                    x[i] = np.clip(x[i] + self.rng.normal(0, 0.012, x[i].shape), 0, 1)
        weights = np.where(targets >= 0.05, 1.0, self.weak_negative_weight).astype(np.float32)
        y = np.concatenate((targets, weights), axis=-1)
        return x[..., None], y


def completed_epochs(history_path: Path) -> int:
    if not history_path.exists():
        return 0
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows)


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
    parser.add_argument("--initial-epoch", type=int, default=-1,
                        help="-1 reads the number of rows already in history.csv")
    parser.add_argument("--cache-shards", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.data / "dataset_summary.json").read_text(encoding="utf-8"))
    weak_weight = (float(args.weak_negative_weight) if args.weak_negative_weight is not None else
                   float(summary["configuration"]["weak_negative_weight"]))
    train = ShardSequence(args.data, "train", args.batch_size, True, weak_weight,
                          20260901, args.cache_shards)
    val = ShardSequence(args.data, "val", args.batch_size, False, weak_weight,
                        20260902, args.cache_shards)
    custom = {"GroupNormalization": GroupNormalization,
              "semantic_prior_v2_loss": semantic_prior_v2_loss,
              "access_mae": access_mae, "motion_mae": motion_mae}
    if args.resume:
        if not args.resume.exists():
            raise FileNotFoundError(args.resume)
        model = tf.keras.models.load_model(args.resume, custom_objects=custom)
    else:
        model = build_semantic_prior_v2(args.base_filters)
    model.compile(tf.keras.optimizers.Adam(args.lr), loss=semantic_prior_v2_loss,
                  metrics=[access_mae, motion_mae])
    history_path = args.output / "history.csv"
    initial_epoch = completed_epochs(history_path) if args.initial_epoch < 0 and args.resume else max(args.initial_epoch, 0)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(args.output / "best_fullframe_semantic_prior_v2.h5"),
                                           monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(args.output / "latest_fullframe_semantic_prior_v2.h5"),
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
    model.save(args.output / "final_restored_best_fullframe_semantic_prior_v2.h5")
    val_losses = history.history.get("val_loss", [])
    report = {
        "initial_epoch": initial_epoch,
        "completed_this_run": len(val_losses),
        "best_epoch_this_run": int(initial_epoch + np.argmin(val_losses) + 1) if val_losses else None,
        "best_val_loss_this_run": float(np.min(val_losses)) if val_losses else None,
        "train_samples": train.sample_count, "val_samples": val.sample_count,
        "train_batches": len(train), "val_batches": len(val),
        "batch_size": args.batch_size, "weak_negative_weight": weak_weight,
    }
    (args.output / "training_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
