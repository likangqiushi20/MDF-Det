"""Build native-resolution full-frame training shards for semantic-prior v2.

"Full-frame" means that crop centres are sampled over the complete WAMI
footprint.  The network still receives native-resolution local patches; a
gigapixel frame is never resized into one network input.  Historical frames
are registered to an anchor frame before their median context is formed.

The labels are local, trailing temporal unions rather than one permanent map:
  * access head: all annotated vehicle trajectories in the label window;
  * motion head: trajectory samples moving at least ``min_move_m``.

Each completed anchor is an independent compressed NPZ shard, so generation
is restartable and does not require keeping the entire dataset in memory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class Homographies:
    """Compose adjacent source->next-frame homographies."""

    def __init__(self, path: Path):
        data = np.load(path)
        self.frames = np.asarray(data["frame_numbers"], np.int64)
        self.matrices = np.asarray(data["matrices"], np.float64)
        self.valid = np.asarray(data["valid"], bool)
        self.image_sizes = np.asarray(data["image_sizes"], np.int32)
        self.index = {int(frame): i for i, frame in enumerate(self.frames)}
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        # Scalar 3x3 multiplication avoids rare Windows BLAS dispatch stalls.
        return np.asarray([[sum(left[r, k] * right[k, c] for k in range(3))
                            for c in range(3)] for r in range(3)], np.float64)

    def compose(self, source: int, destination: int) -> np.ndarray:
        key = int(source), int(destination)
        if key in self._cache:
            return self._cache[key].copy()
        if source == destination:
            result = np.eye(3, dtype=np.float64)
        else:
            si, di = self.index[source], self.index[destination]
            if si > di:
                result = np.linalg.inv(self.compose(destination, source))
            else:
                result = np.eye(3, dtype=np.float64)
                for i in range(si, di):
                    if not self.valid[i]:
                        raise RuntimeError(f"Invalid homography {self.frames[i]}->{self.frames[i + 1]}")
                    result = self._multiply(self.matrices[i], result)
            result = result / result[2, 2]
        self._cache[key] = result
        return result.copy()

    def shape(self, frame: int) -> tuple[int, int]:
        height, width = self.image_sizes[self.index[int(frame)]]
        return int(height), int(width)


def read_gray(frames: Path, frame: int) -> np.ndarray:
    path = frames / f"frame{frame:06d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def moving_rows(table: pd.DataFrame, threshold_m: float) -> pd.DataFrame:
    ordered = table.sort_values(["id", "FRAME_NUMBER"]).copy()
    following = ordered.shift(-1)
    adjacent = ((following["id"] == ordered["id"]) &
                ((following["FRAME_NUMBER"] - ordered["FRAME_NUMBER"]) == 1))
    north = (following["LATITUDE"] - ordered["LATITUDE"]) * 111_320.0
    east = ((following["LONGITUDE"] - ordered["LONGITUDE"]) * 111_320.0 *
            np.cos(np.deg2rad((following["LATITUDE"] + ordered["LATITUDE"]) / 2.0)))
    forward = adjacent & (np.hypot(north, east) >= threshold_m)
    backward = forward.groupby(ordered["id"], sort=False).shift(1, fill_value=False)
    return ordered[forward | backward].copy()


def load_truth(path: Path, min_move_m: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(path)
    required = {"id", "FRAME_NUMBER", "LATITUDE", "LONGITUDE", "X", "Y"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Truth CSV lacks columns: {sorted(missing)}")
    return table, moving_rows(table, min_move_m)


def transform_points(rows: pd.DataFrame, anchor: int, homographies: Homographies) -> np.ndarray:
    """Map per-frame truth X/Y coordinates into the anchor pixel system."""
    groups = []
    for frame, part in rows.groupby("FRAME_NUMBER"):
        frame = int(frame)
        if frame not in homographies.index:
            continue
        xy = part[["X", "Y"]].to_numpy(np.float64)
        homogeneous = np.column_stack((xy, np.ones(len(xy), np.float64)))
        matrix = homographies.compose(frame, anchor)
        mapped = (matrix @ homogeneous.T).T
        mapped = mapped[:, :2] / mapped[:, 2:3]
        groups.append(np.column_stack((mapped[:, 1], mapped[:, 0])))  # row, col
    if not groups:
        return np.empty((0, 2), np.float32)
    result = np.concatenate(groups).astype(np.float32)
    return result[np.isfinite(result).all(axis=1)]


def spatial_fold(row: float, col: float, anchor: int, reference: int,
                 homographies: Homographies, block: int, modulo: int) -> int:
    point = np.asarray([[[col, row]]], np.float64)
    reference_xy = cv2.perspectiveTransform(point, homographies.compose(anchor, reference))[0, 0]
    bx, by = math.floor(reference_xy[0] / block), math.floor(reference_xy[1] / block)
    return int((bx * 73856093 + by * 19349663) % modulo)


def split_accepts(row: float, col: float, split: str, anchor: int, args,
                  homographies: Homographies) -> tuple[bool, int]:
    fold = spatial_fold(row, col, anchor, args.reference_frame, homographies,
                        args.spatial_block, args.val_modulo)
    accepted = fold == args.val_remainder if split == "val" else fold != args.val_remainder
    return accepted, fold


def soft_target(patch: int, points: np.ndarray, top: int, left: int,
                radius: int, sigma: float) -> np.ndarray:
    impulses = np.zeros((patch, patch), np.uint8)
    halo = radius + int(math.ceil(3.0 * sigma))
    for row, col in points:
        rr, cc = int(round(row - top)), int(round(col - left))
        if -halo <= rr < patch + halo and -halo <= cc < patch + halo:
            cv2.circle(impulses, (cc, rr), radius, 1, -1)
    if not impulses.any():
        return np.zeros((patch, patch), np.uint8)
    soft = cv2.GaussianBlur(impulses.astype(np.float32), (0, 0), sigmaX=sigma)
    maximum = float(soft.max())
    if maximum > 0:
        soft /= maximum
    return np.uint8(np.clip(np.rint(soft * 255.0), 0, 255))


def patch_is_covered(source_shape: tuple[int, int], local_matrix: np.ndarray, patch: int) -> bool:
    corners = np.asarray([[[0.0, 0.0], [patch - 1.0, 0.0],
                           [patch - 1.0, patch - 1.0], [0.0, patch - 1.0]]], np.float64)
    source = cv2.perspectiveTransform(corners, np.linalg.inv(local_matrix))[0]
    height, width = source_shape
    return bool((source[:, 0] >= 1).all() and (source[:, 0] < width - 1).all() and
                (source[:, 1] >= 1).all() and (source[:, 1] < height - 1).all())


def context_patch(images: dict[int, np.ndarray], matrices: dict[int, np.ndarray],
                  row: int, col: int, patch: int) -> np.ndarray | None:
    half = patch // 2
    top, left = row - half, col - half
    crop_transform = np.asarray([[1.0, 0.0, -left], [0.0, 1.0, -top], [0.0, 0.0, 1.0]])
    registered = []
    for frame, image in images.items():
        local = crop_transform @ matrices[frame]
        if not patch_is_covered(image.shape, local, patch):
            return None
        warped = cv2.warpPerspective(image, local, (patch, patch), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        registered.append(warped)
    stack = np.stack(registered).astype(np.float32)
    # Remove one global brightness offset per source patch before the median.
    reference_median = np.median(stack[-1])
    for i in range(len(stack) - 1):
        stack[i] = np.clip(stack[i] + reference_median - np.median(stack[i]), 0, 255)
    return np.uint8(np.median(stack, axis=0))


def negative_candidates(count: int, height: int, width: int, half: int,
                        points: np.ndarray, split: str, anchor: int, args,
                        homographies: Homographies, rng: np.random.Generator):
    tree = cKDTree(points) if len(points) else None
    selected: list[tuple[int, int, int]] = []
    attempts = 0
    limit = max(count * 100, 5000)
    while len(selected) < count and attempts < limit:
        attempts += 1
        row = int(rng.integers(half, height - half))
        col = int(rng.integers(half, width - half))
        accepted, fold = split_accepts(row, col, split, anchor, args, homographies)
        if not accepted:
            continue
        if tree is not None and tree.query((row, col), k=1)[0] < args.negative_margin:
            continue
        selected.append((row, col, fold))
    return selected


def texture_score(image: np.ndarray, row: int, col: int, size: int = 64) -> float:
    half = size // 2
    patch = image[row-half:row+half, col-half:col+half]
    if patch.shape != (size, size):
        return -1.0
    laplacian = cv2.Laplacian(patch, cv2.CV_32F)
    return float(laplacian.var() + 0.20 * patch.var())


def choose_anchors(access_points: np.ndarray, current: np.ndarray, split: str, anchor: int,
                   args, homographies: Homographies, rng: np.random.Generator):
    height, width = current.shape
    half = args.patch // 2
    positive_count = int(round(args.samples_per_anchor * args.positive_fraction))
    hard_count = int(round(args.samples_per_anchor * args.hard_negative_fraction))
    random_count = args.samples_per_anchor - positive_count - hard_count
    anchors: list[tuple[int, int, int, int]] = []  # row, col, type, fold

    if len(access_points):
        order = rng.choice(len(access_points), positive_count, replace=len(access_points) < positive_count)
        jitter = min(args.positive_jitter, half - 1)
        for index in order:
            row = int(round(access_points[index, 0] + rng.integers(-jitter, jitter + 1)))
            col = int(round(access_points[index, 1] + rng.integers(-jitter, jitter + 1)))
            if not (half <= row < height - half and half <= col < width - half):
                continue
            accepted, fold = split_accepts(row, col, split, anchor, args, homographies)
            if accepted:
                anchors.append((row, col, 0, fold))

    # Refill positives after fold/border rejection. The positive fraction is a
    # target; it is allowed to be lower if a validation fold contains no GT.
    positive_attempts = 0
    while len(anchors) < positive_count and len(access_points) and positive_attempts < positive_count * 30:
        positive_attempts += 1
        row, col = access_points[int(rng.integers(len(access_points)))]
        row = int(round(row + rng.integers(-args.positive_jitter, args.positive_jitter + 1)))
        col = int(round(col + rng.integers(-args.positive_jitter, args.positive_jitter + 1)))
        if half <= row < height - half and half <= col < width - half:
            accepted, fold = split_accepts(row, col, split, anchor, args, homographies)
            if accepted:
                anchors.append((row, col, 0, fold))

    pool = negative_candidates(max(hard_count * args.hard_pool_multiplier, hard_count),
                               height, width, half, access_points, split, anchor, args,
                               homographies, rng)
    ranked = sorted(pool, key=lambda item: texture_score(current, item[0], item[1]), reverse=True)
    anchors.extend((row, col, 1, fold) for row, col, fold in ranked[:hard_count])
    random_pool = negative_candidates(random_count, height, width, half, access_points,
                                      split, anchor, args, homographies, rng)
    anchors.extend((row, col, 2, fold) for row, col, fold in random_pool[:random_count])
    rng.shuffle(anchors)
    return anchors


def build_anchor(split: str, anchor: int, table: pd.DataFrame, moving: pd.DataFrame,
                 homographies: Homographies, args, rng: np.random.Generator) -> dict:
    label_start = anchor - args.label_history + 1
    access_rows = table[(table.FRAME_NUMBER >= label_start) & (table.FRAME_NUMBER <= anchor)]
    motion_rows = moving[(moving.FRAME_NUMBER >= label_start) & (moving.FRAME_NUMBER <= anchor)]
    access = transform_points(access_rows, anchor, homographies)
    motion = transform_points(motion_rows, anchor, homographies)
    height, width = homographies.shape(anchor)
    inside = lambda pts: pts[(pts[:, 0] >= 0) & (pts[:, 0] < height) &
                             (pts[:, 1] >= 0) & (pts[:, 1] < width)] if len(pts) else pts
    access, motion = inside(access), inside(motion)

    history_frames = list(range(anchor - args.history, anchor))
    images = {frame: read_gray(args.frames, frame) for frame in history_frames}
    current = read_gray(args.frames, anchor)
    matrices = {frame: homographies.compose(frame, anchor) for frame in history_frames}
    anchors = choose_anchors(access, current, split, anchor, args, homographies, rng)

    x_values, y_values, metadata = [], [], []
    for row, col, sample_type, fold in anchors:
        context = context_patch(images, matrices, row, col, args.patch)
        if context is None:
            continue
        top, left = row - args.patch // 2, col - args.patch // 2
        access_target = soft_target(args.patch, access, top, left,
                                    args.positive_radius, args.gaussian_sigma)
        motion_target = soft_target(args.patch, motion, top, left,
                                    args.positive_radius, args.gaussian_sigma)
        x_values.append(context)
        y_values.append(np.stack((access_target, motion_target), axis=-1))
        metadata.append((anchor, row, col, sample_type, fold))

    if not x_values:
        raise RuntimeError(f"Anchor {anchor} produced no fully covered samples")
    split_dir = args.output / split
    split_dir.mkdir(parents=True, exist_ok=True)
    destination = split_dir / f"anchor_{anchor:06d}.npz"
    saver = np.savez if args.no_compress else np.savez_compressed
    saver(destination, images=np.asarray(x_values, np.uint8), targets=np.asarray(y_values, np.uint8),
          metadata=np.asarray(metadata, np.int32))
    types = np.asarray(metadata)[:, 3]
    return {
        "anchor": anchor, "samples": len(metadata), "access_points": len(access),
        "motion_points": len(motion), "positive": int((types == 0).sum()),
        "hard_negative": int((types == 1).sum()), "random_negative": int((types == 2).sum()),
        "path": str(destination.resolve()),
    }


def anchor_range(start: int, stop: int, step: int, history: int):
    first = max(start, start + history)
    return list(range(first, stop + 1, step))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--homographies", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-start", type=int, default=100)
    parser.add_argument("--train-stop", type=int, default=379)
    parser.add_argument("--val-start", type=int, default=400)
    parser.add_argument("--val-stop", type=int, default=499)
    parser.add_argument("--anchor-step", type=int, default=5)
    parser.add_argument("--history", type=int, default=5, help="Registered context frames before each anchor")
    parser.add_argument("--label-history", type=int, default=20, help="Trailing frames forming each local trajectory label")
    parser.add_argument("--patch", type=int, default=384)
    parser.add_argument("--samples-per-anchor", type=int, default=96)
    parser.add_argument("--positive-fraction", type=float, default=0.40)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.30)
    parser.add_argument("--hard-pool-multiplier", type=int, default=8)
    parser.add_argument("--positive-jitter", type=int, default=96)
    parser.add_argument("--positive-radius", type=int, default=10)
    parser.add_argument("--gaussian-sigma", type=float, default=5.0)
    parser.add_argument("--negative-margin", type=float, default=48.0)
    parser.add_argument("--min-move-m", type=float, default=0.8)
    parser.add_argument("--weak-negative-weight", type=float, default=0.03)
    parser.add_argument("--reference-frame", type=int, default=100)
    parser.add_argument("--spatial-block", type=int, default=2048)
    parser.add_argument("--val-modulo", type=int, default=5)
    parser.add_argument("--val-remainder", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    args = parser.parse_args()
    if args.patch <= 0 or args.patch % 8:
        raise ValueError("--patch must be a positive multiple of 8")
    if args.train_stop >= args.val_start:
        raise ValueError("Training and validation frame ranges must be disjoint")
    if args.positive_fraction + args.hard_negative_fraction > 1.0:
        raise ValueError("positive_fraction + hard_negative_fraction must not exceed 1")
    args.output.mkdir(parents=True, exist_ok=True)
    homographies = Homographies(args.homographies)
    table, moving = load_truth(args.truth_csv, args.min_move_m)

    summaries = {"train": [], "val": []}
    for split, start, stop in (("train", args.train_start, args.train_stop),
                               ("val", args.val_start, args.val_stop)):
        frames = anchor_range(start, stop, args.anchor_step, args.history)
        for position, anchor in enumerate(frames, 1):
            destination = args.output / split / f"anchor_{anchor:06d}.npz"
            if destination.exists() and not args.overwrite:
                with np.load(destination) as saved:
                    count = int(len(saved["images"]))
                    types = saved["metadata"][:, 3]
                result = {"anchor": anchor, "samples": count,
                          "positive": int((types == 0).sum()),
                          "hard_negative": int((types == 1).sum()),
                          "random_negative": int((types == 2).sum()),
                          "path": str(destination.resolve()), "resumed": True}
            else:
                local_rng = np.random.default_rng(args.seed + anchor + (100000 if split == "val" else 0))
                result = build_anchor(split, anchor, table, moving, homographies, args, local_rng)
            summaries[split].append(result)
            print(f"{split} [{position}/{len(frames)}] anchor={anchor}: "
                  f"samples={result['samples']} pos/hard/random="
                  f"{result['positive']}/{result['hard_negative']}/{result['random_negative']}", flush=True)

    configuration = {key: str(value.resolve()) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    report = {
        "design": "full-footprint native crops; registered median history; trailing trajectory labels; no coordinates",
        "configuration": configuration,
        "train_shards": len(summaries["train"]),
        "val_shards": len(summaries["val"]),
        "train_samples": int(sum(item["samples"] for item in summaries["train"])),
        "val_samples": int(sum(item["samples"] for item in summaries["val"])),
        "splits": summaries,
    }
    (args.output / "dataset_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("train_shards", "val_shards", "train_samples", "val_samples")},
                     indent=2))


if __name__ == "__main__":
    main()
