"""Build moving/stationary classified truth on HM-Net's r1 AOI grids."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from common.data.fixed_grid_truth import project_truth_points
from common.labels.truth import TruthPolicy, iter_truth_tracks


def build_cache(manifest_path: Path, grids_path: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixed = json.loads(grids_path.read_text(encoding="utf-8"))
    grids = fixed["papers"]["hm_net"]["grids"]
    dataset_root = Path(manifest["dataset_root"])
    moving_policy = TruthPolicy(name="canonical_adjacent_any_motion")
    all_policy = TruthPolicy(name="all_objects", moving_only=False)
    columns: dict[str, list] = {
        name: []
        for name in ("split", "aoi", "frame", "track_id", "x", "y", "class_id")
    }
    split_codes = {"train": 0, "self_test": 1}
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for split, split_code in split_codes.items():
        counts[split] = {}
        relative = manifest["frame_sets"][split]["truth_files"]["r1"]
        for track in iter_truth_tracks(dataset_root / relative):
            selected = all_policy.apply(track)
            moving_keys = {
                (point.track_id, point.frame_number)
                for point in moving_policy.apply(track)
            }
            for aoi, grid in grids.items():
                for item in project_truth_points(selected, grid):
                    key = (item.source.track_id, item.source.frame_number)
                    class_id = 0 if key in moving_keys else 1
                    columns["split"].append(split_code)
                    columns["aoi"].append(int(aoi))
                    columns["frame"].append(item.source.frame_number)
                    columns["track_id"].append(item.source.track_id)
                    columns["x"].append(item.x)
                    columns["y"].append(item.y)
                    columns["class_id"].append(class_id)
                    class_name = "moving" if class_id == 0 else "stationary"
                    aoi_counts = counts[split].setdefault(
                        aoi, {"moving": 0, "stationary": 0}
                    )
                    aoi_counts[class_name] += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        split=np.asarray(columns["split"], dtype=np.uint8),
        aoi=np.asarray(columns["aoi"], dtype=np.uint8),
        frame=np.asarray(columns["frame"], dtype=np.uint16),
        track_id=np.asarray(columns["track_id"], dtype=np.uint32),
        x=np.asarray(columns["x"], dtype=np.float32),
        y=np.asarray(columns["y"], dtype=np.float32),
        class_id=np.asarray(columns["class_id"], dtype=np.uint8),
    )
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cache": str(output_path.resolve()),
        "records": len(columns["frame"]),
        "classes": {"moving": 0, "stationary": 1},
        "classification": "canonical_adjacent_any_motion_0.8m_else_stationary",
        "counts": counts,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
