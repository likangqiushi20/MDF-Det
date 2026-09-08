"""Measure how often CATLoss localization patches contain multiple targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=float, default=45.0)
    args = parser.parse_args()
    data = np.load(args.truth_cache)
    result = {}
    half = args.patch_size / 2
    for aoi in sorted(np.unique(data["aoi"][data["split"] == 0])):
        split_mask = (data["split"] == 0) & (data["aoi"] == aoi)
        frames = data["frame"][split_mask]
        xs = data["x"][split_mask]
        ys = data["y"][split_mask]
        counts = []
        for frame in np.unique(frames):
            mask = frames == frame
            points = np.column_stack((xs[mask], ys[mask]))
            if not len(points):
                continue
            inside = (
                (np.abs(points[:, None, 0] - points[None, :, 0]) <= half)
                & (np.abs(points[:, None, 1] - points[None, :, 1]) <= half)
            )
            counts.extend(inside.sum(axis=1).tolist())
        values = np.asarray(counts)
        result[f"{int(aoi):02d}"] = {
            "patches": len(values),
            "r_ge_2": int((values >= 2).sum()),
            "r_ge_3": int((values >= 3).sum()),
            "fraction_r_ge_2": float((values >= 2).mean()),
            "maximum_r": int(values.max()) if len(values) else 0,
        }
    output = {"schema_version": 1, "split": "train", "aois": result}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
