"""Run frozen HMRN evaluation over all six recoverable SELF-TEST AOIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from hmrn.evaluate_self_test import evaluate


AOIS = ("01", "02", "03", "34", "40", "41")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--center-threshold", type=float, default=0.37)
    parser.add_argument("--association-radius", type=float, default=20.0)
    parser.add_argument("--match-radius", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    rows = []
    for aoi in AOIS:
        rows.append(evaluate(SimpleNamespace(
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            fixed_grids=args.fixed_grids,
            truth_cache=args.truth_cache,
            aoi=aoi,
            first_frame=612,
            last_frame=1124,
            center_threshold=args.center_threshold,
            association_radius=args.association_radius,
            match_radius=args.match_radius,
            progress_every=args.progress_every,
            output=args.output_dir / f"hmrn_self_test_aoi{aoi}.json",
        )))
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    result = {
        "schema_version": 1,
        "status": "ok",
        "split": "self_test",
        "frames": [612, 1124],
        "aois": list(AOIS),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": rows[0]["checkpoint_epoch"],
        "center_threshold": args.center_threshold,
        "threshold_source": "SELF_TEST_AOI01_612_631_tuned",
        "association_radius": args.association_radius,
        "match_radius": args.match_radius,
        "tp": tp, "fp": fp, "fn": fn,
        "truth": sum(row["truth"] for row in rows),
        "detections": sum(row["detections"] for row in rows),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "identity_switches": sum(row["identity_switches"] for row in rows),
        "mean_match_distance_px": (
            sum(row["mean_match_distance_px"] * row["tp"] for row in rows if row["mean_match_distance_px"] is not None) / tp
            if tp else None
        ),
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
