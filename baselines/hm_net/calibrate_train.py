"""Calibrate HM-Net confidence threshold on TRAIN validation frames only."""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50],
    )
    parser.add_argument("--max-detections-per-class", type=int, default=2000)
    parser.add_argument("--sgr-threshold", type=float, default=0.28)
    parser.add_argument("--sgr-detection-threshold", type=float, default=0.32)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda:0")
    model = HMNet().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    totals = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in args.thresholds
    }
    peak_scores = []
    started = time.perf_counter()
    with HMNetData(args.manifest, args.fixed_grids, args.truth_cache) as data:
        for aoi in ("01", "02", "03", "34", "40", "41"):
            previous_image = data.read_frame("train", 560, aoi)
            height, width = previous_image.shape
            feedback = torch.zeros(1, 2, height, width, device=device)
            for frame in range(561, 612):
                current_image = data.read_frame("train", frame, aoi)
                current, previous, padded_feedback = _pad_inputs(
                    _image_tensor(current_image, device),
                    _image_tensor(previous_image, device),
                    feedback,
                )
                with torch.no_grad():
                    prediction = model(current, previous, padded_feedback)
                    prediction = {
                        name: values[..., :height, :width]
                        for name, values in prediction.items()
                    }
                    candidates = [
                        item
                        for item in decode(
                            prediction["center"],
                            prediction["motion"],
                            prediction["precision"],
                            threshold=min(args.thresholds),
                            max_detections_per_class=(
                                args.max_detections_per_class
                            ),
                        )[0]
                        if item.class_id == 0
                    ]
                    feedback = selective_gaussian_reconstruction(
                        prediction["center"],
                        filter_threshold=args.sgr_threshold,
                        detection_threshold=args.sgr_detection_threshold,
                        amplification=1.2,
                    )
                peak_scores.extend(item.score for item in candidates)
                truths = [
                    point for point in data.points("train", frame, aoi)
                    if point.class_id == 0
                ]
                for threshold in args.thresholds:
                    selected = [
                        item for item in candidates if item.score >= threshold
                    ]
                    matched = match_points(
                        [(item.x, item.y) for item in selected],
                        [(item.x, item.y) for item in truths],
                        10,
                    )
                    totals[threshold]["tp"] += matched.true_positives
                    totals[threshold]["fp"] += matched.false_positives
                    totals[threshold]["fn"] += matched.false_negatives
                previous_image = current_image
            print(json.dumps({"completed_aoi": aoi}), flush=True)
    sweep = []
    for threshold in args.thresholds:
        row = {"threshold": threshold, **totals[threshold]}
        row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0
        row["recall"] = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0
        row["f1"] = (
            2 * row["precision"] * row["recall"]
            / (row["precision"] + row["recall"])
            if row["precision"] + row["recall"]
            else 0
        )
        sweep.append(row)
    best = max(sweep, key=lambda row: row["f1"])
    result = {
        "schema_version": 1,
        "status": "ok",
        "split": "train",
        "frames": [561, 611],
        "aois": ["01", "02", "03", "34", "40", "41"],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "sgr_threshold": args.sgr_threshold,
        "sgr_detection_threshold": args.sgr_detection_threshold,
        "score_quantiles": (
            {
                str(quantile): float(np.quantile(peak_scores, quantile))
                for quantile in (0, 0.25, 0.5, 0.75, 0.9, 0.99, 1)
            }
            if peak_scores
            else {}
        ),
        "sweep": sweep,
        "selected": best,
        "selection_source": "TRAIN_validation_only",
        "elapsed_seconds": time.perf_counter() - started,
        "self_test_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
