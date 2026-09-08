"""Audit geographic AOI bounds against sampled WPAFB raster headers."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import rasterio
from rasterio.windows import Window, bounds as window_bounds, from_bounds


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_m * math.asin(math.sqrt(value))


def _round_window(window: Window) -> Window:
    col_start = math.floor(window.col_off)
    row_start = math.floor(window.row_off)
    col_end = math.ceil(window.col_off + window.width)
    row_end = math.ceil(window.row_off + window.height)
    return Window(
        col_off=col_start,
        row_off=row_start,
        width=col_end - col_start,
        height=row_end - row_start,
    )


def _edge_error_m(requested: dict[str, float], actual: tuple[float, ...]) -> float:
    left, bottom, right, top = actual
    center_lat = (requested["bottom"] + requested["top"]) / 2.0
    center_lon = (requested["left"] + requested["right"]) / 2.0
    errors = (
        _haversine_m(center_lat, requested["left"], center_lat, left),
        _haversine_m(center_lat, requested["right"], center_lat, right),
        _haversine_m(requested["bottom"], center_lon, bottom, center_lon),
        _haversine_m(requested["top"], center_lon, top, center_lon),
    )
    return max(errors)


def audit_aoi_on_raster(path: Path, bounds: dict[str, float]) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        raw_window = from_bounds(
            bounds["left"],
            bounds["bottom"],
            bounds["right"],
            bounds["top"],
            transform=dataset.transform,
        )
        rounded = _round_window(raw_window)
        actual_bounds = window_bounds(rounded, dataset.transform)
        raw_right = raw_window.col_off + raw_window.width
        raw_bottom = raw_window.row_off + raw_window.height
        rounded_right = rounded.col_off + rounded.width
        rounded_bottom = rounded.row_off + rounded.height
        max_edge_error_px = max(
            abs(raw_window.col_off - rounded.col_off),
            abs(raw_window.row_off - rounded.row_off),
            abs(raw_right - rounded_right),
            abs(raw_bottom - rounded_bottom),
        )
        inside = (
            rounded.col_off >= 0
            and rounded.row_off >= 0
            and rounded.col_off + rounded.width <= dataset.width
            and rounded.row_off + rounded.height <= dataset.height
        )
        return {
            "raster_size": [dataset.width, dataset.height],
            "window": {
                "col_off": int(rounded.col_off),
                "row_off": int(rounded.row_off),
                "width": int(rounded.width),
                "height": int(rounded.height),
            },
            "inside_raster": inside,
            "rounded_geographic_bounds": {
                "left": actual_bounds[0],
                "bottom": actual_bounds[1],
                "right": actual_bounds[2],
                "top": actual_bounds[3],
            },
            "max_rounded_edge_error_m": _edge_error_m(bounds, actual_bounds),
            "max_rounded_edge_error_px": max_edge_error_px,
        }


def _sample_frames(
    frames: list[dict[str, Any]], frame_set: str, level: str
) -> list[dict[str, Any]]:
    candidates = [
        frame
        for frame in frames
        if frame["set"] == frame_set and level in frame["levels"]
    ]
    if not candidates:
        return []
    indexes = sorted({0, len(candidates) // 2, len(candidates) - 1})
    return [candidates[index] for index in indexes]


def build_aoi_audit(
    manifest_path: Path,
    profiles_path: Path,
    output_path: Path,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    dataset_root = Path(manifest["dataset_root"])
    geographic = profiles["aoi_profiles"]["geographic_bounds_wgs84"]
    levels = sorted(
        {
            profile["level"]
            for profile in profiles["resolution_profiles"].values()
        }
    )

    records: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for frame_set in ("train", "self_test"):
        for level in levels:
            for frame in _sample_frames(manifest["frames"], frame_set, level):
                raster_path = dataset_root / frame["levels"][level]["path"]
                for aoi_name, bounds in geographic.items():
                    if "left" not in bounds:
                        continue
                    result = audit_aoi_on_raster(raster_path, bounds)
                    record = {
                        "set": frame_set,
                        "level": level,
                        "frame_number": frame["frame_number"],
                        "aoi": aoi_name,
                        "raster_path": frame["levels"][level]["path"],
                        **result,
                    }
                    records.append(record)
                    summaries[(frame_set, level, aoi_name)].append(result)

    summary_rows = []
    for (frame_set, level, aoi_name), results in sorted(summaries.items()):
        widths = [result["window"]["width"] for result in results]
        heights = [result["window"]["height"] for result in results]
        edge_errors = [result["max_rounded_edge_error_m"] for result in results]
        pixel_edge_errors = [
            result["max_rounded_edge_error_px"] for result in results
        ]
        summary_rows.append(
            {
                "set": frame_set,
                "level": level,
                "aoi": aoi_name,
                "sample_count": len(results),
                "all_inside_raster": all(result["inside_raster"] for result in results),
                "width_range": [min(widths), max(widths)],
                "height_range": [min(heights), max(heights)],
                "max_rounded_edge_error_m": max(edge_errors),
                "max_rounded_edge_error_px": max(pixel_edge_errors),
            }
        )

    output = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "profiles": str(profiles_path.resolve()),
        "unresolved_aois": [
            name for name, bounds in geographic.items() if "left" not in bounds
        ],
        "summary": summary_rows,
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
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = build_aoi_audit(args.manifest, args.profiles, args.output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
