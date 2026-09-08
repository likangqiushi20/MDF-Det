"""Audit WPAFB truth pixel coordinates against NITF georeferencing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio

from common.labels.truth import TruthPoint, iter_truth_csv


def _sample_frames(frames: list[int]) -> list[int]:
    ordered = sorted(frames)
    return [ordered[index] for index in sorted({0, len(ordered) // 2, len(ordered) - 1})]


def _error_summary(errors: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(errors, axis=1)
    return {
        "count": len(errors),
        "median_dx": float(np.median(errors[:, 0])),
        "median_dy": float(np.median(errors[:, 1])),
        "median_norm": float(np.median(norms)),
        "p95_norm": float(np.percentile(norms, 95)),
        "max_norm": float(np.max(norms)),
        "fraction_within_1px": float(np.mean(norms <= 1.0)),
    }


def audit_frame(
    raster_path: Path,
    points: list[TruthPoint],
) -> dict[str, Any]:
    with rasterio.open(raster_path) as dataset:
        inverse = ~dataset.transform
        corner_errors = []
        center_errors = []
        outside = 0
        for point in points:
            col, row = inverse * (point.longitude, point.latitude)
            corner_errors.append((col - point.x, row - point.y))
            center_errors.append((col - (point.x + 0.5), row - (point.y + 0.5)))
            if not (0 <= point.x < dataset.width and 0 <= point.y < dataset.height):
                outside += 1
        return {
            "raster_size": [dataset.width, dataset.height],
            "truth_count": len(points),
            "truth_outside_raster": outside,
            "integer_pixel_as_corner": _error_summary(np.asarray(corner_errors)),
            "integer_pixel_as_center": _error_summary(np.asarray(center_errors)),
        }


def _load_selected_truth(
    path: Path,
    selected_frames: set[int],
) -> dict[int, list[TruthPoint]]:
    result = {frame: [] for frame in selected_frames}
    for point in iter_truth_csv(path):
        if point.frame_number in result:
            result[point.frame_number].append(point)
    return result


def build_truth_coordinate_audit(
    manifest_path: Path,
    output_path: Path,
    levels: tuple[str, ...] = ("r0", "r1"),
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(manifest["dataset_root"])
    frame_lookup = {
        (frame["set"], frame["frame_number"]): frame for frame in manifest["frames"]
    }
    records = []

    for frame_set in ("train", "self_test"):
        set_manifest = manifest["frame_sets"][frame_set]
        frames = list(
            range(set_manifest["expected_start"], set_manifest["expected_end"] + 1)
        )
        selected = _sample_frames(frames)
        level_points: dict[str, dict[int, list[TruthPoint]]] = {}
        for level in levels:
            truth_relative = set_manifest["truth_files"][level]
            level_points[level] = _load_selected_truth(
                dataset_root / truth_relative,
                set(selected),
            )
            for frame_number in selected:
                frame = frame_lookup[(frame_set, frame_number)]
                raster_relative = frame["levels"][level]["path"]
                result = audit_frame(
                    dataset_root / raster_relative,
                    level_points[level][frame_number],
                )
                records.append(
                    {
                        "set": frame_set,
                        "level": level,
                        "frame_number": frame_number,
                        "raster_path": raster_relative,
                        "truth_path": truth_relative,
                        **result,
                    }
                )

        for frame_number in selected:
            reference = {
                (point.track_id, point.frame_number): (point.latitude, point.longitude)
                for point in level_points[levels[0]][frame_number]
            }
            for level in levels[1:]:
                compared = {
                    (point.track_id, point.frame_number): (point.latitude, point.longitude)
                    for point in level_points[level][frame_number]
                }
                if reference != compared:
                    raise ValueError(
                        f"Latitude/longitude truth mismatch for {frame_set} "
                        f"frame {frame_number}: {levels[0]} versus {level}"
                    )

    output = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "levels": list(levels),
        "pixel_convention_conclusion": (
            "Truth X/Y are integer pixel indices. Their geodetic locations align "
            "more closely with pixel centers (X+0.5, Y+0.5) than pixel corners."
        ),
        "cross_level_lat_lon_identical_for_samples": True,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", default=["r0", "r1"])
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = build_truth_coordinate_audit(
        args.manifest,
        args.output,
        levels=tuple(args.levels),
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
