"""Build fixed north-up WGS84 AOI grids for each paper profile."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import rasterio
from affine import Affine
from rasterio.transform import xy


EARTH_RADIUS_M = 6_371_008.8


def _half_size(size: list[int]) -> list[int]:
    return [(value + 1) // 2 for value in size]


def _nominal_gsd_by_level(resolution_audit: dict[str, Any]) -> dict[str, float]:
    result = {}
    for level in ("r0", "r1", "r2", "r3", "r4", "r5"):
        values = []
        for frame_set in ("train", "self_test"):
            for sample in resolution_audit["sets"][frame_set][level]["samples"]:
                values.append(sample["meters_per_pixel"]["mean"])
        result[level] = median(values)
    return result


def _grid_from_center(
    center_lon: float,
    center_lat: float,
    width: int,
    height: int,
    gsd_m: float,
) -> dict[str, Any]:
    lat_deg_per_pixel = math.degrees(gsd_m / EARTH_RADIUS_M)
    lon_deg_per_pixel = math.degrees(
        gsd_m / (EARTH_RADIUS_M * math.cos(math.radians(center_lat)))
    )
    left = center_lon - width * lon_deg_per_pixel / 2.0
    top = center_lat + height * lat_deg_per_pixel / 2.0
    transform = Affine(
        lon_deg_per_pixel,
        0.0,
        left,
        0.0,
        -lat_deg_per_pixel,
        top,
    )
    right = left + width * lon_deg_per_pixel
    bottom = top - height * lat_deg_per_pixel
    return {
        "crs": "EPSG:4326",
        "width": width,
        "height": height,
        "nominal_gsd_m": gsd_m,
        "transform": list(transform)[:6],
        "bounds": {
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top,
        },
        "center": {"longitude": center_lon, "latitude": center_lat},
    }


def build_fixed_grid_profiles(
    paper_profiles_path: Path,
    resolution_audit_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> Path:
    profiles = json.loads(paper_profiles_path.read_text(encoding="utf-8"))
    resolution_audit = json.loads(resolution_audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(manifest["dataset_root"])
    geographic = profiles["aoi_profiles"]["geographic_bounds_wgs84"]
    gsd_by_level = _nominal_gsd_by_level(resolution_audit)

    cluster_sizes = profiles["aoi_profiles"][
        "clusternet_reported_full_resolution_sizes"
    ]
    medium_boxes = profiles["aoi_profiles"]["hmrn_hm_net_r1_pixel_boxes"]
    target_sizes = {
        "clusternet": cluster_sizes,
        "catloss": {
            name: _half_size(size) for name, size in cluster_sizes.items()
        },
        "hmrn": {name: box["size"] for name, box in medium_boxes.items()},
        "hm_net": {name: box["size"] for name, box in medium_boxes.items()},
    }
    frame_100 = next(
        frame
        for frame in manifest["frames"]
        if frame["set"] == "train" and frame["frame_number"] == 100
    )
    reference_relative = frame_100["levels"]["r1"]["path"]
    with rasterio.open(dataset_root / reference_relative) as reference:
        if (reference.width, reference.height) != (13312, 10752):
            raise ValueError(
                "HMRN/HM-Net paper centers require the initial 13312x10752 r1 "
                f"canvas, got {reference.width}x{reference.height}"
            )
        medium_centers = {
            name: xy(
                reference.transform,
                box["center"][1],
                box["center"][0],
                offset="center",
            )
            for name, box in medium_boxes.items()
        }

    paper_grids = {}
    for paper_name, sizes in target_sizes.items():
        level = profiles["resolution_profiles"][paper_name]["level"]
        grids = {}
        for aoi_name, size in sizes.items():
            if paper_name in {"hmrn", "hm_net"}:
                center_lon, center_lat = medium_centers[aoi_name]
                center_basis = (
                    "paper pixel center transformed through frame 100 r1 "
                    "13312x10752 georeferencing"
                )
            else:
                bounds = geographic[aoi_name]
                center_lon = (bounds["left"] + bounds["right"]) / 2.0
                center_lat = (bounds["bottom"] + bounds["top"]) / 2.0
                center_basis = "center of legacy geographic AOI bounds"
            grids[aoi_name] = _grid_from_center(
                center_lon,
                center_lat,
                int(size[0]),
                int(size[1]),
                gsd_by_level[level],
            )
        paper_grids[paper_name] = {
            "level": level,
            "grid_status": "provisional_center_verified_size_verified",
            "center_basis": center_basis,
            "size_basis": (
                "paper-reported size"
                if paper_name != "catloss"
                else "half of paper full-resolution AOI size, rounded up"
            ),
            "grids": grids,
        }

    output = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "paper_profiles": str(paper_profiles_path.resolve()),
        "resolution_audit": str(resolution_audit_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "medium_resolution_center_reference": {
            "frame": 100,
            "level": "r1",
            "size": [13312, 10752],
            "path": reference_relative,
        },
        "nominal_gsd_m_by_level": gsd_by_level,
        "warning": (
            "ClusterNet/CATLoss centers come from legacy geographic bounds. "
            "HMRN/HM-Net centers come from paper pixel coordinates on the "
            "initial r1 canvas. Visual validation is still required before L3 claims."
        ),
        "papers": paper_grids,
        "excluded": {
            "aoi_42": "No exact geographic center or boundary is available locally."
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-profiles", type=Path, required=True)
    parser.add_argument("--resolution-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = build_fixed_grid_profiles(
        args.paper_profiles,
        args.resolution_audit,
        args.manifest,
        args.output,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
