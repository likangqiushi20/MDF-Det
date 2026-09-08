"""Infer a full-resolution semantic vehicle prior on an AOI sequence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

try:
    from .model import GroupNormalization
except ImportError:  # Allow: python SPGF/infer.py ...
    from model import GroupNormalization


def context_image(frames: Path, start: int, stop: int, step: int):
    images = []
    for frame in range(start, stop + 1, step):
        image = cv2.imread(str(frames / f"frame{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(frames / f"frame{frame:06d}.png")
        images.append(image)
    return np.median(np.stack(images), axis=0).astype(np.uint8)


def tiled_predict(model, image, tile, overlap):
    height, width = image.shape
    stride = tile - overlap
    rows = list(range(0, max(height - tile, 0) + 1, stride))
    cols = list(range(0, max(width - tile, 0) + 1, stride))
    if not rows or rows[-1] != height - tile: rows.append(max(height - tile, 0))
    if not cols or cols[-1] != width - tile: cols.append(max(width - tile, 0))
    total = np.zeros((height, width, 2), np.float32)
    weight = np.zeros((height, width, 1), np.float32)
    window_1d = np.hanning(tile).astype(np.float32)
    window = np.maximum(np.outer(window_1d, window_1d), .05)[..., None]
    for row in rows:
        for col in cols:
            patch = image[row:row+tile, col:col+tile]
            prediction = model.predict(patch[None, ..., None].astype(np.float32) / 255.0, verbose=0)[0]
            total[row:row+tile, col:col+tile] += prediction * window
            weight[row:row+tile, col:col+tile] += window
    return total / np.maximum(weight, 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-start", type=int, required=True)
    parser.add_argument("--context-stop", type=int, required=True)
    parser.add_argument("--context-step", type=int, default=1)
    parser.add_argument("--mode", choices=("full", "tiled"), default="full")
    parser.add_argument("--tile", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=96)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    context = context_image(args.frames, args.context_start, args.context_stop, args.context_step)
    model = tf.keras.models.load_model(args.model,
                                       custom_objects={"GroupNormalization": GroupNormalization}, compile=False)
    if args.mode == "full":
        prediction = model.predict(context[None, ..., None].astype(np.float32) / 255.0, verbose=0)[0]
    else:
        prediction = tiled_predict(model, context, args.tile, args.overlap)
    access, motion = prediction[..., 0], prediction[..., 1]
    combined = np.sqrt(np.clip(access * motion, 0, 1))
    np.savez_compressed(args.output / "semantic_prior_v2.npz", access=access, motion=motion, combined=combined)
    # Plain arrays can be consumed directly by the published detector's
    # --static-prior-map option without loading this training module.
    np.save(args.output / "access_prior.npy", access.astype(np.float32))
    np.save(args.output / "motion_prior.npy", motion.astype(np.float32))
    np.save(args.output / "combined_prior.npy", combined.astype(np.float32))
    cv2.imwrite(str(args.output / "context.png"), context)
    cv2.imwrite(str(args.output / "access_prior.png"), np.uint8(access * 255))
    cv2.imwrite(str(args.output / "motion_prior.png"), np.uint8(motion * 255))
    cv2.imwrite(str(args.output / "combined_prior.png"), np.uint8(combined * 255))
    report = {"shape": list(context.shape), "mode": args.mode,
              "access_mean": float(access.mean()), "motion_mean": float(motion.mean()),
              "combined_mean": float(combined.mean())}
    (args.output / "inference_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
