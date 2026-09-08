"""Build leakage-controlled scene-prior V4 training shards.

V4 keeps the native-resolution, coordinate-free V2 network input, but removes
all supervision and image crops from the six publication AOIs.  An exclusion
rectangle is obtained for each AOI in reference-frame pixels from a robust
geographic-to-pixel homography fitted using truth points *outside* the held-out
AOIs.  The rectangles are propagated to every anchor by the precomputed frame
homographies.

For every anchor:
  * GT inside an AOI plus ``exclusion_margin`` is removed before labels exist;
  * a crop centre is rejected if its complete patch would touch that protected
    region;
  * train and validation obey the same exclusion rule;
  * saved metadata are audited before the shard is committed.

The conservative axis-aligned rectangles can exclude slightly more background
than the geographic quadrilaterals.  They cannot admit an overlapping crop.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from . import build_dataset_utils as v2
except ImportError:  # Allow: python SPGF/build_dataset.py ...
    import build_dataset_utils as v2


AOI_GEO_BOUNDS = {
    "01": {"ulx": -84.125664, "uly": 39.771185, "lrx": -84.119895, "lry": 39.765354},
    "02": {"ulx": -84.122165, "uly": 39.786161, "lrx": -84.116273, "lry": 39.780145},
    "03": {"ulx": -84.101297, "uly": 39.781737, "lrx": -84.095836, "lry": 39.775665},
    "34": {"ulx": -84.126152, "uly": 39.783150, "lrx": -84.114983, "lry": 39.776215},
    "40": {"ulx": -84.127380, "uly": 39.772167, "lrx": -84.118790, "lry": 39.765719},
    "41": {"ulx": -84.121796, "uly": 39.772351, "lrx": -84.113391, "lry": 39.764676},
}


def inside_any_geo_aoi(table: pd.DataFrame) -> np.ndarray:
    inside = np.zeros(len(table), bool)
    lon, lat = table["LONGITUDE"].to_numpy(), table["LATITUDE"].to_numpy()
    for bounds in AOI_GEO_BOUNDS.values():
        inside |= ((lon >= min(bounds["ulx"], bounds["lrx"])) &
                   (lon <= max(bounds["ulx"], bounds["lrx"])) &
                   (lat >= min(bounds["uly"], bounds["lry"])) &
                   (lat <= max(bounds["uly"], bounds["lry"])))
    return inside


@dataclass
class GeographicCalibration:
    centre: np.ndarray
    scale: np.ndarray
    matrix: np.ndarray
    rmse_px: float
    median_error_px: float
    p95_error_px: float
    points: int
    inlier_ratio: float

    def project(self, lon_lat: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(lon_lat, np.float64) - self.centre) / self.scale
        return cv2.perspectiveTransform(normalized[None], self.matrix)[0]


def fit_geographic_calibration(table: pd.DataFrame, reference_frame: int) -> GeographicCalibration:
    reference = table[table.FRAME_NUMBER == reference_frame].copy()
    reference = reference[~inside_any_geo_aoi(reference)]
    reference = reference.drop_duplicates(["LONGITUDE", "LATITUDE"])
    if len(reference) < 20:
        raise RuntimeError("Too few non-AOI points to calibrate geographic AOI boundaries")
    source = reference[["LONGITUDE", "LATITUDE"]].to_numpy(np.float64)
    destination = reference[["X", "Y"]].to_numpy(np.float64)
    centre, scale = source.mean(axis=0), source.std(axis=0)
    normalized = (source - centre) / scale
    matrix, inliers = cv2.findHomography(normalized, destination, cv2.RANSAC, 3.0)
    if matrix is None:
        raise RuntimeError("Geographic-to-reference homography fitting failed")
    predicted = cv2.perspectiveTransform(normalized[None], matrix)[0]
    error = np.linalg.norm(predicted - destination, axis=1)
    calibration = GeographicCalibration(
        centre=centre, scale=scale, matrix=matrix,
        rmse_px=float(np.sqrt(np.mean(error ** 2))),
        median_error_px=float(np.median(error)),
        p95_error_px=float(np.quantile(error, 0.95)),
        points=len(reference), inlier_ratio=float(inliers.mean()),
    )
    if calibration.p95_error_px > 5.0:
        raise RuntimeError(f"AOI geographic calibration is unreliable: p95={calibration.p95_error_px:.2f}px")
    return calibration


class SpatialExclusion:
    def __init__(self, calibration: GeographicCalibration, homographies: v2.Homographies,
                 reference_frame: int, margin: int, patch: int):
        self.homographies = homographies
        self.reference_frame = int(reference_frame)
        self.margin = int(margin)
        self.patch = int(patch)
        self.reference_polygons = {}
        for aoi, bounds in AOI_GEO_BOUNDS.items():
            corners = np.asarray([
                [bounds["ulx"], bounds["uly"]], [bounds["lrx"], bounds["uly"]],
                [bounds["lrx"], bounds["lry"]], [bounds["ulx"], bounds["lry"]],
            ], np.float64)
            self.reference_polygons[aoi] = calibration.project(corners)
        self._cache = {}

    def boxes(self, anchor: int, for_centres: bool = False) -> dict[str, tuple[float, float, float, float]]:
        key = int(anchor), bool(for_centres)
        if key in self._cache:
            return self._cache[key]
        matrix = self.homographies.compose(self.reference_frame, anchor)
        padding = self.margin + (self.patch // 2 if for_centres else 0)
        output = {}
        for aoi, polygon in self.reference_polygons.items():
            mapped = cv2.perspectiveTransform(polygon[None].astype(np.float64), matrix)[0]
            output[aoi] = (float(mapped[:, 1].min() - padding),
                           float(mapped[:, 1].max() + padding),
                           float(mapped[:, 0].min() - padding),
                           float(mapped[:, 0].max() + padding))  # row0,row1,col0,col1
        self._cache[key] = output
        return output

    @staticmethod
    def point_allowed(row: float, col: float, boxes) -> bool:
        return not any(row0 <= row <= row1 and col0 <= col <= col1
                       for row0, row1, col0, col1 in boxes.values())

    def filter_points(self, points: np.ndarray, anchor: int) -> tuple[np.ndarray, int]:
        if not len(points):
            return points, 0
        boxes = self.boxes(anchor, for_centres=False)
        keep = np.asarray([self.point_allowed(row, col, boxes) for row, col in points], bool)
        return points[keep], int((~keep).sum())

    def centre_allowed(self, row: float, col: float, anchor: int) -> bool:
        return self.point_allowed(row, col, self.boxes(anchor, for_centres=True))


def negative_candidates(count, height, width, half, points, split, anchor, args,
                        homographies, exclusion, rng):
    from scipy.spatial import cKDTree
    tree = cKDTree(points) if len(points) else None
    selected, attempts, rejected_aoi = [], 0, 0
    limit = max(count * 150, 7500)
    while len(selected) < count and attempts < limit:
        attempts += 1
        row = int(rng.integers(half, height - half))
        col = int(rng.integers(half, width - half))
        if not exclusion.centre_allowed(row, col, anchor):
            rejected_aoi += 1
            continue
        accepted, fold = v2.split_accepts(row, col, split, anchor, args, homographies)
        if not accepted:
            continue
        if tree is not None and tree.query((row, col), k=1)[0] < args.negative_margin:
            continue
        selected.append((row, col, fold))
    return selected, rejected_aoi


def choose_anchors(access_points, current, split, anchor, args, homographies, exclusion, rng):
    height, width = current.shape
    half = args.patch // 2
    positive_count = int(round(args.samples_per_anchor * args.positive_fraction))
    hard_count = int(round(args.samples_per_anchor * args.hard_negative_fraction))
    random_count = args.samples_per_anchor - positive_count - hard_count
    anchors, rejected_aoi = [], 0

    attempts = 0
    while len(anchors) < positive_count and len(access_points) and attempts < positive_count * 100:
        attempts += 1
        point = access_points[int(rng.integers(len(access_points)))]
        row = int(round(point[0] + rng.integers(-args.positive_jitter, args.positive_jitter + 1)))
        col = int(round(point[1] + rng.integers(-args.positive_jitter, args.positive_jitter + 1)))
        if not (half <= row < height - half and half <= col < width - half):
            continue
        if not exclusion.centre_allowed(row, col, anchor):
            rejected_aoi += 1
            continue
        accepted, fold = v2.split_accepts(row, col, split, anchor, args, homographies)
        if accepted:
            anchors.append((row, col, 0, fold))

    pool, rejected = negative_candidates(max(hard_count * args.hard_pool_multiplier, hard_count),
                                         height, width, half, access_points, split, anchor,
                                         args, homographies, exclusion, rng)
    rejected_aoi += rejected
    ranked = sorted(pool, key=lambda item: v2.texture_score(current, item[0], item[1]), reverse=True)
    anchors.extend((row, col, 1, fold) for row, col, fold in ranked[:hard_count])
    random_pool, rejected = negative_candidates(random_count, height, width, half, access_points,
                                                split, anchor, args, homographies, exclusion, rng)
    rejected_aoi += rejected
    anchors.extend((row, col, 2, fold) for row, col, fold in random_pool[:random_count])
    rng.shuffle(anchors)
    return anchors, rejected_aoi


def build_anchor(split, anchor, table, moving, homographies, exclusion, args, rng):
    label_start = anchor - args.label_history + 1
    access_rows = table[(table.FRAME_NUMBER >= label_start) & (table.FRAME_NUMBER <= anchor)]
    motion_rows = moving[(moving.FRAME_NUMBER >= label_start) & (moving.FRAME_NUMBER <= anchor)]
    access = v2.transform_points(access_rows, anchor, homographies)
    motion = v2.transform_points(motion_rows, anchor, homographies)
    height, width = homographies.shape(anchor)
    inside = lambda pts: pts[(pts[:, 0] >= 0) & (pts[:, 0] < height) &
                             (pts[:, 1] >= 0) & (pts[:, 1] < width)] if len(pts) else pts
    access, motion = inside(access), inside(motion)
    access, removed_access = exclusion.filter_points(access, anchor)
    motion, removed_motion = exclusion.filter_points(motion, anchor)

    history_frames = list(range(anchor - args.history, anchor))
    images = {frame: v2.read_gray(args.frames, frame) for frame in history_frames}
    current = v2.read_gray(args.frames, anchor)
    matrices = {frame: homographies.compose(frame, anchor) for frame in history_frames}
    anchors, rejected_aoi = choose_anchors(access, current, split, anchor, args,
                                           homographies, exclusion, rng)
    x_values, y_values, metadata = [], [], []
    for row, col, sample_type, fold in anchors:
        if not exclusion.centre_allowed(row, col, anchor):
            raise AssertionError(f"AOI-overlapping sample escaped rejection at anchor {anchor}")
        context = v2.context_patch(images, matrices, row, col, args.patch)
        if context is None:
            continue
        top, left = row - args.patch // 2, col - args.patch // 2
        access_target = v2.soft_target(args.patch, access, top, left,
                                       args.positive_radius, args.gaussian_sigma)
        motion_target = v2.soft_target(args.patch, motion, top, left,
                                       args.positive_radius, args.gaussian_sigma)
        x_values.append(context)
        y_values.append(np.stack((access_target, motion_target), axis=-1))
        metadata.append((anchor, row, col, sample_type, fold))
    if not x_values:
        raise RuntimeError(f"Anchor {anchor} produced no leakage-free samples")
    centre_boxes = np.asarray(list(exclusion.boxes(anchor, True).values()), np.float32)
    for _, row, col, _, _ in metadata:
        if any(r0 <= row <= r1 and c0 <= col <= c1 for r0, r1, c0, c1 in centre_boxes):
            raise AssertionError("Saved sample metadata failed V4 AOI-overlap audit")
    destination = args.output / split / f"anchor_{anchor:06d}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez if args.no_compress else np.savez_compressed
    saver(destination, images=np.asarray(x_values, np.uint8),
          targets=np.asarray(y_values, np.uint8), metadata=np.asarray(metadata, np.int32),
          excluded_centre_boxes=centre_boxes)
    types = np.asarray(metadata)[:, 3]
    return {"anchor": anchor, "samples": len(metadata), "access_points": len(access),
            "motion_points": len(motion), "removed_access_points": removed_access,
            "removed_motion_points": removed_motion, "aoi_candidate_rejections": rejected_aoi,
            "positive": int((types == 0).sum()), "hard_negative": int((types == 1).sum()),
            "random_negative": int((types == 2).sum()), "audit_passed": True,
            "path": str(destination.resolve())}


def save_exclusion_preview(args, homographies, exclusion):
    image = v2.read_gray(args.frames, args.reference_frame)
    scale = min(1.0, 2000.0 / max(image.shape))
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    for aoi, polygon in exclusion.reference_polygons.items():
        points = np.rint(polygon * scale).astype(np.int32)
        cv2.polylines(preview, [points], True, (0, 0, 255), 3)
        x, y = points[0]
        cv2.putText(preview, f"AOI{aoi}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.output / "excluded_aois_reference.png"), preview)


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
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--label-history", type=int, default=20)
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
    parser.add_argument("--exclusion-margin", type=int, default=64,
                        help="Extra protected pixels outside each AOI; patch half-size is added automatically")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    args = parser.parse_args()
    if args.patch <= 0 or args.patch % 8:
        raise ValueError("--patch must be a positive multiple of 8")
    if args.exclusion_margin < args.positive_radius + int(np.ceil(3 * args.gaussian_sigma)):
        raise ValueError("--exclusion-margin must cover positive_radius + 3*gaussian_sigma")
    if args.train_stop >= args.val_start:
        raise ValueError("Training and validation frame ranges must be disjoint")
    args.output.mkdir(parents=True, exist_ok=True)
    homographies = v2.Homographies(args.homographies)
    raw_table, _ = v2.load_truth(args.truth_csv, args.min_move_m)
    calibration = fit_geographic_calibration(raw_table, args.reference_frame)
    exclusion = SpatialExclusion(calibration, homographies, args.reference_frame,
                                 args.exclusion_margin, args.patch)
    save_exclusion_preview(args, homographies, exclusion)

    # Remove exact geographic AOI annotations before either target head is built.
    geo_excluded = inside_any_geo_aoi(raw_table)
    table = raw_table[~geo_excluded].copy()
    moving = v2.moving_rows(table, args.min_move_m)
    summaries = {"train": [], "val": []}
    for split, start, stop in (("train", args.train_start, args.train_stop),
                               ("val", args.val_start, args.val_stop)):
        frames = v2.anchor_range(start, stop, args.anchor_step, args.history)
        for position, anchor in enumerate(frames, 1):
            destination = args.output / split / f"anchor_{anchor:06d}.npz"
            if destination.exists() and not args.overwrite:
                with np.load(destination) as saved:
                    metadata = saved["metadata"]
                    boxes = saved["excluded_centre_boxes"]
                    for _, row, col, _, _ in metadata:
                        if any(r0 <= row <= r1 and c0 <= col <= c1 for r0, r1, c0, c1 in boxes):
                            raise AssertionError(f"Existing shard {destination} failed V4 audit")
                    types = metadata[:, 3]
                result = {"anchor": anchor, "samples": len(metadata),
                          "positive": int((types == 0).sum()),
                          "hard_negative": int((types == 1).sum()),
                          "random_negative": int((types == 2).sum()),
                          "audit_passed": True, "resumed": True,
                          "path": str(destination.resolve())}
            else:
                rng = np.random.default_rng(args.seed + anchor + (100000 if split == "val" else 0))
                result = build_anchor(split, anchor, table, moving, homographies,
                                      exclusion, args, rng)
            summaries[split].append(result)
            print(f"{split} [{position}/{len(frames)}] anchor={anchor}: samples={result['samples']} "
                  f"pos/hard/random={result['positive']}/{result['hard_negative']}/"
                  f"{result['random_negative']} audit=PASS", flush=True)

    configuration = {key: str(value.resolve()) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    report = {
        "version": "SPGF-V4",
        "design": "native crops with six publication AOIs, labels, overlap windows and safety margins excluded",
        "held_out_aois": list(AOI_GEO_BOUNDS),
        "aoi_geo_bounds": AOI_GEO_BOUNDS,
        "configuration": configuration,
        "geographic_calibration": {
            "points": calibration.points, "inlier_ratio": calibration.inlier_ratio,
            "rmse_px": calibration.rmse_px, "median_error_px": calibration.median_error_px,
            "p95_error_px": calibration.p95_error_px,
            "fit_uses_held_out_aoi_truth": False,
        },
        "truth_rows_before_exclusion": int(len(raw_table)),
        "truth_rows_removed_by_geo_aoi": int(geo_excluded.sum()),
        "truth_rows_after_exclusion": int(len(table)),
        "effective_centre_padding_px": args.exclusion_margin + args.patch // 2,
        "train_shards": len(summaries["train"]), "val_shards": len(summaries["val"]),
        "train_samples": int(sum(item["samples"] for item in summaries["train"])),
        "val_samples": int(sum(item["samples"] for item in summaries["val"])),
        "all_shards_audit_passed": all(item.get("audit_passed", False)
                                         for split in summaries.values() for item in split),
        "splits": summaries,
    }
    (args.output / "dataset_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "held_out_aois", "truth_rows_removed_by_geo_aoi", "effective_centre_padding_px",
        "train_shards", "val_shards", "train_samples", "val_samples", "all_shards_audit_passed")},
        indent=2))


if __name__ == "__main__":
    main()
