"""Evaluate a frozen HM-Net checkpoint on one full SELF-TEST AOI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from common.metrics.point_matching import match_points
from hm_net.data import HMNetData
from hm_net.infer import _image_tensor, _pad_inputs
from hm_net.model import HMNet
from hm_net.sgr import selective_gaussian_reconstruction
from hm_net.tracking import decode


def evaluate(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda:0")
    model = HMNet().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    frame_rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "truth": 0, "detections": 0}
    feedback = None
    previous_image = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with HMNetData(args.manifest, args.fixed_grids, args.truth_cache) as data:
        for frame in range(args.first_frame, args.last_frame + 1):
            frame_started = time.perf_counter()
            current_image = data.read_frame("self_test", frame, args.aoi)
            height, width = current_image.shape
            if previous_image is None:
                previous_image = current_image
                feedback = torch.zeros(1, 2, height, width, device=device)
            assert feedback is not None
            current_tensor, previous_tensor, padded_feedback = _pad_inputs(
                _image_tensor(current_image, device),
                _image_tensor(previous_image, device),
                feedback,
            )
            with torch.no_grad():
                prediction = model(
                    current_tensor, previous_tensor, padded_feedback
                )
                prediction = {
                    name: values[..., :height, :width]
                    for name, values in prediction.items()
                }
                all_detections = decode(
                    prediction["center"],
                    prediction["motion"],
                    prediction["precision"],
                    threshold=args.detection_threshold,
                )[0]
                feedback = selective_gaussian_reconstruction(
                    prediction["center"],
                    filter_threshold=args.sgr_threshold,
                    detection_threshold=args.detection_threshold,
                    amplification=args.sgr_amplification,
                )
            moving = [
                detection for detection in all_detections
                if detection.class_id == 0
            ]
            truths = [
                point for point in data.points("self_test", frame, args.aoi)
                if point.class_id == 0
            ]
            matched = match_points(
                [(item.x, item.y) for item in moving],
                [(item.x, item.y) for item in truths],
                args.match_radius,
            )
            row = {
                "frame": frame,
                "detections": len(moving),
                "truth": len(truths),
                "tp": matched.true_positives,
                "fp": matched.false_positives,
                "fn": matched.false_negatives,
                "seconds": time.perf_counter() - frame_started,
            }
            frame_rows.append(row)
            for key in totals:
                totals[key] += row[key]
            previous_image = current_image
            if len(frame_rows) % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "aoi": args.aoi,
                            "frame": frame,
                            "completed": len(frame_rows),
                            "total": args.last_frame - args.first_frame + 1,
                            **totals,
                        }
                    ),
                    flush=True,
                )
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": "self_test",
        "aoi": args.aoi,
        "frames": [args.first_frame, args.last_frame],
        "thresholds": {
            "detection": args.detection_threshold,
            "sgr": args.sgr_threshold,
            "sgr_amplification": args.sgr_amplification,
            "match_radius": args.match_radius,
            "source": "paper_frozen_not_test_tuned",
        },
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "mean_frame_seconds": float(np.mean([row["seconds"] for row in frame_rows])),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "per_frame": frame_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "per_frame"}))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--aoi", required=True)
    parser.add_argument("--first-frame", type=int, default=612)
    parser.add_argument("--last-frame", type=int, default=1124)
    parser.add_argument("--detection-threshold", type=float, default=0.32)
    parser.add_argument("--sgr-threshold", type=float, default=0.28)
    parser.add_argument("--sgr-amplification", type=float, default=1.2)
    parser.add_argument("--match-radius", type=float, default=10)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
