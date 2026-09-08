"""Audit the shared r1 AOI, cache and optimized training contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def audit(
    protocol_path: Path,
    grids_path: Path,
    cache_path: Path,
    source_fixed_grids_path: Path | None = None,
) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    fixed = json.loads(grids_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    papers = protocol["target_papers"]
    reference = fixed["papers"][papers[0]]
    identical = all(
        fixed["papers"][paper]["level"] == protocol["image_level"]
        and fixed["papers"][paper]["grids"] == reference["grids"]
        for paper in papers
    )
    source_grid_matches = True
    if source_fixed_grids_path is not None:
        source_fixed = json.loads(source_fixed_grids_path.read_text(encoding="utf-8"))
        source_grid_matches = (
            source_fixed["papers"]["hm_net"]["level"] == protocol["image_level"]
            and source_fixed["papers"]["hm_net"]["grids"] == reference["grids"]
        )
    cache_shapes_match = True
    for aoi in protocol["aois"]:
        item = cache["aois"].get(aoi)
        grid = reference["grids"][aoi]
        if item is None or item["shape"] != [512, grid["height"], grid["width"]]:
            cache_shapes_match = False
            break
        array = np.load(cache_path.parent / item["file"], mmap_mode="r")
        if array.dtype != np.uint8 or list(array.shape) != item["shape"]:
            cache_shapes_match = False
            break
    expected_fingerprint = hashlib.sha256(
        json.dumps(
            {"level": reference["level"], "grids": reference["grids"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cache_fingerprint_matches = cache.get("grid_sha256") == expected_fingerprint
    gsd = float(reference["grids"][protocol["aois"][0]]["nominal_gsd_m"])
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if identical and cache_shapes_match and source_grid_matches and cache_fingerprint_matches else "fail",
        "comparison_name": protocol["name"],
        "papers": papers,
        "image_level": protocol["image_level"],
        "aois": protocol["aois"],
        "all_method_grids_identical": identical,
        "shared_cache_shapes_match": cache_shapes_match,
        "legacy_cache_source_grid_matches": source_grid_matches,
        "cache_grid_fingerprint_matches": cache_fingerprint_matches,
        "grid_sha256": expected_fingerprint,
        "cache_frames": cache["frames"],
        "nominal_gsd_m_per_pixel": gsd,
        "match_radius_m": protocol["evaluation"]["match_radius_m"],
        "match_radius_px": protocol["evaluation"]["match_radius_m"] / gsd,
    }
    if result["status"] != "pass":
        raise ValueError(json.dumps(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--source-fixed-grids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.protocol, args.fixed_grids, args.frame_cache, args.source_fixed_grids
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
