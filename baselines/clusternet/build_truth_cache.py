"""Build r0 fixed-grid ClusterNet truth with persistent and five-frame flags."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from clusternet.truth import five_frame_motion_keys
from common.data.fixed_grid_truth import project_truth_points
from common.labels.truth import iter_truth_tracks


def build_cache(manifest_path: Path, grids_path: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixed = json.loads(grids_path.read_text(encoding="utf-8"))
    profile = fixed["papers"]["clusternet"]
    grids = profile["grids"]
    truth_level = profile["level"]
    root = Path(manifest["dataset_root"])
    columns = {name: [] for name in ("split", "aoi", "frame", "track_id", "x", "y", "moving")}
    counts = {}
    for split, split_code in {"train": 0, "self_test": 1}.items():
        counts[split] = {}
        truth_path = root / manifest["frame_sets"][split]["truth_files"][truth_level]
        for track in iter_truth_tracks(truth_path):
            moving = five_frame_motion_keys(track)
            for aoi, grid in grids.items():
                projected = project_truth_points(track, grid)
                for item in projected:
                    point = item.source
                    columns["split"].append(split_code)
                    columns["aoi"].append(int(aoi))
                    columns["frame"].append(point.frame_number)
                    columns["track_id"].append(point.track_id)
                    columns["x"].append(item.x)
                    columns["y"].append(item.y)
                    is_moving = (point.track_id, point.frame_number) in moving
                    columns["moving"].append(is_moving)
                    entry = counts[split].setdefault(aoi, {"persistent": 0, "moving": 0})
                    entry["persistent"] += 1
                    entry["moving"] += int(is_moving)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        split=np.asarray(columns["split"], dtype=np.uint8),
        aoi=np.asarray(columns["aoi"], dtype=np.uint8),
        frame=np.asarray(columns["frame"], dtype=np.uint16),
        track_id=np.asarray(columns["track_id"], dtype=np.uint32),
        x=np.asarray(columns["x"], dtype=np.float32),
        y=np.asarray(columns["y"], dtype=np.float32),
        moving=np.asarray(columns["moving"], dtype=np.bool_),
    )
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "records": len(columns["frame"]),
        "cache": str(output_path.resolve()),
        "policy": "five_consecutive_frames_displacement_ge_3.75m",
        "truth_level": truth_level,
        "counts": counts,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_cache(args.manifest, args.fixed_grids, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
