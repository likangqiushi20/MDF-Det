"""Run frozen HM-Net evaluation over all six recoverable SELF-TEST AOIs."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from hm_net.evaluate_self_test import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--detection-threshold", type=float, default=0.16)
    args = parser.parse_args()
    reports = []
    for aoi in ("01", "02", "03", "34", "40", "41"):
        print(json.dumps({"starting_aoi": aoi}), flush=True)
        report = evaluate(
            Namespace(
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                fixed_grids=args.fixed_grids,
                truth_cache=args.truth_cache,
                aoi=aoi,
                first_frame=612,
                last_frame=1124,
                detection_threshold=args.detection_threshold,
                sgr_threshold=0.28,
                sgr_amplification=1.2,
                match_radius=10.0,
                progress_every=25,
                output=args.output_dir / f"hmnet_self_test_aoi{aoi}.json",
            )
        )
        reports.append({key: value for key, value in report.items() if key != "per_frame"})
    totals = {
        key: sum(report[key] for report in reports)
        for key in ("tp", "fp", "fn", "truth", "detections", "elapsed_seconds")
    }
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "split": "self_test",
        "frames": [612, 1124],
        "aois": [report["aoi"] for report in reports],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": reports[0]["checkpoint_epoch"],
        "detection_threshold": args.detection_threshold,
        "threshold_source": "TRAIN_561_611_six_AOI_calibration",
        "sgr_threshold": 0.28,
        "sgr_amplification": 1.2,
        "match_radius": 10,
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "per_aoi": reports,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
