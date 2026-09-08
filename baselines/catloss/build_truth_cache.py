"""Build a compact per-frame moving-truth cache for CATLoss fixed grids."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from common.data.fixed_grid_truth import project_truth_points
from common.labels.truth import TruthPolicy, iter_truth_tracks


def build_cache(manifest_path: Path, grids_path: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixed = json.loads(grids_path.read_text(encoding="utf-8"))
    grids = fixed["papers"]["catloss"]["grids"]
    dataset_root = Path(manifest["dataset_root"])
    policy = TruthPolicy(name="canonical_adjacent_any_motion")

    columns: dict[str, list] = {
        "split": [],
        "aoi": [],
        "frame": [],
        "track_id": [],
        "x": [],
        "y": [],
    }
    split_codes = {"train": 0, "self_test": 1}
    counts: dict[str, dict[str, int]] = {}
    for split, split_code in split_codes.items():
        counts[split] = {}
        relative = manifest["frame_sets"][split]["truth_files"]["r1"]
        for track in iter_truth_tracks(dataset_root / relative):
            moving = policy.apply(track)
            if not moving:
                continue
            for aoi, grid in grids.items():
                projected = project_truth_points(moving, grid)
                for item in projected:
                    columns["split"].append(split_code)
                    columns["aoi"].append(int(aoi))
                    columns["frame"].append(item.source.frame_number)
                    columns["track_id"].append(item.source.track_id)
                    columns["x"].append(item.x)
                    columns["y"].append(item.y)
                    counts[split][aoi] = counts[split].get(aoi, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        split=np.asarray(columns["split"], dtype=np.uint8),
        aoi=np.asarray(columns["aoi"], dtype=np.uint8),
        frame=np.asarray(columns["frame"], dtype=np.uint16),
        track_id=np.asarray(columns["track_id"], dtype=np.uint32),
        x=np.asarray(columns["x"], dtype=np.float32),
        y=np.asarray(columns["y"], dtype=np.float32),
    )
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cache": str(output_path.resolve()),
        "records": len(columns["frame"]),
        "split_codes": split_codes,
        "policy": "canonical_adjacent_any_motion_0.8m_consecutive_frames",
        "counts": counts,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_cache(args.manifest, args.fixed_grids, args.output),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
