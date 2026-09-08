"""Read small previews from real NITFs through every unique fixed AOI grid."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def _sample_frames(frames: list[dict[str, Any]], frame_set: str, level: str) -> list[dict[str, Any]]:
    candidates = [
        frame for frame in frames if frame["set"] == frame_set and level in frame["levels"]
    ]
    indexes = sorted({0, len(candidates) // 2, len(candidates) - 1})
    return [candidates[index] for index in indexes]


def _unique_grids(fixed_grids: dict[str, Any]) -> list[dict[str, Any]]:
    unique = {}
    for paper_name, paper in fixed_grids["papers"].items():
        for aoi_name, grid in paper["grids"].items():
            key = (
                paper["level"],
                aoi_name,
                grid["width"],
                grid["height"],
                tuple(grid["transform"]),
            )
            item = unique.setdefault(
                key,
                {
                    "level": paper["level"],
                    "aoi": aoi_name,
                    "grid": grid,
                    "papers": [],
                },
            )
            item["papers"].append(paper_name)
    return list(unique.values())


def build_fixed_grid_smoke(
    manifest_path: Path,
    fixed_grids_path: Path,
    output_path: Path,
    preview_size: int = 64,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixed_grids = json.loads(fixed_grids_path.read_text(encoding="utf-8"))
    dataset_root = Path(manifest["dataset_root"])
    records = []

    for item in _unique_grids(fixed_grids):
        level = item["level"]
        grid = item["grid"]
        transform = Affine(*grid["transform"])
        for frame_set in ("train", "self_test"):
            for frame in _sample_frames(manifest["frames"], frame_set, level):
                relative = frame["levels"][level]["path"]
                started = time.perf_counter()
                with rasterio.open(dataset_root / relative) as source:
                    with WarpedVRT(
                        source,
                        crs=grid["crs"],
                        transform=transform,
                        width=grid["width"],
                        height=grid["height"],
                        resampling=Resampling.bilinear,
                        nodata=0,
                    ) as vrt:
                        preview = vrt.read(
                            1,
                            out_shape=(preview_size, preview_size),
                            resampling=Resampling.bilinear,
                            masked=True,
                        )
                elapsed = time.perf_counter() - started
                valid_count = int(np.ma.count(preview))
                records.append(
                    {
                        "set": frame_set,
                        "frame_number": frame["frame_number"],
                        "level": level,
                        "aoi": item["aoi"],
                        "papers": sorted(item["papers"]),
                        "source_path": relative,
                        "target_size": [grid["width"], grid["height"]],
                        "preview_size": [preview_size, preview_size],
                        "valid_fraction": valid_count / preview.size,
                        "minimum": float(preview.min()) if valid_count else None,
                        "maximum": float(preview.max()) if valid_count else None,
                        "mean": float(preview.mean()) if valid_count else None,
                        "elapsed_seconds": elapsed,
                    }
                )

    output = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "fixed_grids": str(fixed_grids_path.resolve()),
        "all_previews_have_valid_pixels": all(
            record["valid_fraction"] > 0 for record in records
        ),
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
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-size", type=int, default=64)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = build_fixed_grid_smoke(
        args.manifest,
        args.fixed_grids,
        args.output,
        preview_size=args.preview_size,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
