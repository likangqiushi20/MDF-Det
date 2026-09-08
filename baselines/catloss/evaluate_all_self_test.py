"""Evaluate frozen CATLoss thresholds over six SELF-TEST AOIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from catloss.evaluate_self_test import evaluate


AOIS = ("01", "02", "03", "34", "40", "41")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--foreground-quantile", type=float, default=0.975)
    parser.add_argument("--objectness-threshold", type=float, default=0.70)
    parser.add_argument("--complex-area-threshold", type=int, default=200)
    parser.add_argument("--localization-peak-threshold", type=float, default=0.40)
    parser.add_argument("--match-radius", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    rows = []
    for aoi in AOIS:
        rows.append(evaluate(SimpleNamespace(
            **vars(args), aoi=aoi, first_frame=612, last_frame=1124,
            output=args.output_dir / f"catloss_self_test_aoi{aoi}.json",
        )))
    tp, fp, fn = (sum(row[name] for row in rows) for name in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    result = {
        "schema_version": 1, "status": "ok", "split": "self_test",
        "frames": [612, 1124], "aois": list(AOIS),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": rows[0]["checkpoint_epoch"],
        "thresholds": rows[0]["thresholds"],
        "tp": tp, "fp": fp, "fn": fn,
        "truth": sum(row["truth"] for row in rows),
        "detections": sum(row["detections"] for row in rows),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
        "gpu_peak_gib": max(row["gpu_peak_gib"] for row in rows),
        "per_aoi": rows,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
