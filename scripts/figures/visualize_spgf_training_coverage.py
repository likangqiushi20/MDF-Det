"""Visualize SPGF training-crop coverage and audit overlap with held-out AOIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SPGF.build_dataset_utils import Homographies, read_gray
from SPGF.build_dataset import fit_geographic_calibration, SpatialExclusion


TRAIN_COLOR = (178, 114, 0)    # colour-blind-safe blue, BGR (#0072B2)
VAL_COLOR = (0, 159, 230)      # colour-blind-safe orange, BGR (#E69F00)
TEST_COLOR = (0, 69, 204)      # vermilion/red, BGR (#CC4500)


def transform_polygon(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points_xy[None].astype(np.float64), matrix)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-max-side", type=int, default=2600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.dataset / "dataset_summary.json").read_text(encoding="utf-8"))
    config = summary["configuration"]
    frames = Path(config["frames"])
    truth_csv = Path(config["truth_csv"])
    reference = int(config["reference_frame"])
    patch = int(config["patch"])
    homographies = Homographies(Path(config["homographies"]))

    raw_truth = pd.read_csv(truth_csv)
    calibration = fit_geographic_calibration(raw_truth, reference)
    exclusion = SpatialExclusion(
        calibration, homographies, reference,
        int(config["exclusion_margin"]), patch,
    )
    reference_image = read_gray(frames, reference)
    height, width = reference_image.shape
    scale = min(1.0, args.preview_max_side / max(height, width))
    preview_size = (int(round(width * scale)), int(round(height * scale)))
    background = cv2.resize(reference_image, preview_size, interpolation=cv2.INTER_AREA)
    base = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    split_masks = {
        split: np.zeros((preview_size[1], preview_size[0]), np.uint8)
        for split in ("train", "val")
    }
    sample_count = {"train": 0, "val": 0}
    overlap_count = {"train": 0, "val": 0}
    maximum_overlap_area = 0.0
    overlap_by_aoi = {
        split: {aoi: 0 for aoi in exclusion.reference_polygons}
        for split in ("train", "val")
    }
    half = patch / 2.0

    for split in ("train", "val"):
        for shard in sorted((args.dataset / split).glob("*.npz")):
            with np.load(shard) as data:
                metadata = np.asarray(data["metadata"])
            for anchor, row, col, _sample_type, _fold in metadata:
                corners = np.asarray([
                    [col - half, row - half], [col + half, row - half],
                    [col + half, row + half], [col - half, row + half],
                ], np.float64)
                crop_ref = transform_polygon(
                    corners, homographies.compose(int(anchor), reference)
                )
                preview_polygon = np.rint(crop_ref * scale).astype(np.int32)
                cv2.fillConvexPoly(split_masks[split], preview_polygon, 255)
                sample_count[split] += 1
                for aoi, aoi_polygon in exclusion.reference_polygons.items():
                    area, _ = cv2.intersectConvexConvex(
                        crop_ref.astype(np.float32), aoi_polygon.astype(np.float32)
                    )
                    if area > 1e-3:
                        overlap_count[split] += 1
                        overlap_by_aoi[split][aoi] += 1
                        maximum_overlap_area = max(maximum_overlap_area, float(area))

    aoi_pairwise_overlaps = []
    aoi_names = list(exclusion.reference_polygons)
    for left_index, left_name in enumerate(aoi_names):
        left_polygon = exclusion.reference_polygons[left_name].astype(np.float32)
        left_area = abs(float(cv2.contourArea(left_polygon)))
        for right_name in aoi_names[left_index + 1:]:
            right_polygon = exclusion.reference_polygons[right_name].astype(np.float32)
            right_area = abs(float(cv2.contourArea(right_polygon)))
            area, _ = cv2.intersectConvexConvex(left_polygon, right_polygon)
            area = float(area)
            if area > 1e-3:
                aoi_pairwise_overlaps.append({
                    "aoi_pair": [left_name, right_name],
                    "intersection_area_px2": area,
                    "fraction_of_first_aoi": area / max(left_area, 1e-12),
                    "fraction_of_second_aoi": area / max(right_area, 1e-12),
                })

    # Retain only the actual reference-frame WAMI footprint.  Its exterior is
    # white (and cropped away as far as a rectangular paper figure permits),
    # rather than being shown as an uninformative black surround.
    # This export uses both 0 and near-white (250--255) as exterior fill along
    # different tiled edges.  Build the footprint from radiometrically valid
    # pixels so neither fill value appears in the publication figure.
    valid_full = ((reference_image > 0) & (reference_image < 250)).astype(np.uint8)
    contour_result = cv2.findContours(
        valid_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contour_result[-2]
    footprint = np.zeros_like(valid_full)
    cv2.drawContours(footprint, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    valid = cv2.resize(footprint, preview_size, interpolation=cv2.INTER_NEAREST) > 0
    overlay = np.full_like(base, 255)
    overlay[valid] = base[valid]
    for split, color in (("train", TRAIN_COLOR), ("val", VAL_COLOR)):
        mask = (split_masks[split] > 0) & valid
        overlay[mask] = (
            0.66 * overlay[mask] + 0.34 * np.asarray(color)
        ).astype(np.uint8)

    # AOI polygons are red; labels are positioned near their top-left vertices.
    for aoi, polygon in exclusion.reference_polygons.items():
        points = np.rint(polygon * scale).astype(np.int32)
        cv2.polylines(overlay, [points], True, TEST_COLOR, 7, cv2.LINE_AA)
        x, y = points[np.argmin(points[:, 0] + points[:, 1])]
        cv2.putText(overlay, f"AOI {aoi}", (int(x), max(42, int(y) - 11)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.12, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(overlay, f"AOI {aoi}", (int(x), max(42, int(y) - 11)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.12, TEST_COLOR, 3, cv2.LINE_AA)

    ys, xs = np.nonzero(valid)
    margin = 12
    y0_crop, y1_crop = max(0, int(ys.min()) - margin), min(overlay.shape[0], int(ys.max()) + margin + 1)
    x0_crop, x1_crop = max(0, int(xs.min()) - margin), min(overlay.shape[1], int(xs.max()) + margin + 1)
    overlay = overlay[y0_crop:y1_crop, x0_crop:x1_crop]

    legend = [
        (TRAIN_COLOR, "Training regions"),
        (VAL_COLOR, "Validation regions"),
        (TEST_COLOR, "Held-out AOI test regions"),
    ]
    x0, y0 = 43, 62
    cv2.rectangle(overlay, (18, 13), (690, 188), (255, 255, 255), -1)
    cv2.rectangle(overlay, (18, 13), (690, 188), (90, 90, 90), 3)
    for index, (color, text) in enumerate(legend):
        y = y0 + index * 52
        cv2.rectangle(overlay, (x0, y - 25), (x0 + 38, y + 13), color, -1)
        cv2.putText(overlay, text, (x0 + 56, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.98, (20, 20, 20), 2, cv2.LINE_AA)

    output_image = args.output / "spgf_v4_train_validation_test_regions_paper.png"
    cv2.imwrite(str(output_image), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    report = {
        "reference_frame": reference,
        "training_samples": sample_count["train"],
        "validation_samples": sample_count["val"],
        "patch_size": patch,
        "held_out_aois": list(exclusion.reference_polygons),
        "training_crops_intersecting_any_aoi": overlap_count["train"],
        "validation_crops_intersecting_any_aoi": overlap_count["val"],
        "overlap_by_aoi": overlap_by_aoi,
        "maximum_intersection_area_px2": maximum_overlap_area,
        "overlap_audit_passed": all(value == 0 for value in overlap_count.values()),
        "pairwise_overlaps_among_held_out_aois": aoi_pairwise_overlaps,
        "preview_scale": scale,
        "output_image": str(output_image.resolve()),
    }
    (args.output / "spgf_training_overlap_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
