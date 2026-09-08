"""Calibrate CATLoss localization routing and peak thresholds on TRAIN validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from catloss.data import crop_stack, valid_center
from catloss.heatmaps import extract_peaks
from catloss.model import LocalizationNetwork, ObjectnessNetwork
from catloss.postprocess import extract_foreground_regions
from catloss.real_data import CATLossData
from common.metrics.point_matching import match_points


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    device = torch.device("cuda:0")
    object_model = ObjectnessNetwork(cfg["temporal_channels"]).to(device).eval()
    local_model = LocalizationNetwork(
        input_channels=cfg["temporal_channels"],
        dilation_mode=cfg["dilation_mode"],
    ).to(device).eval()
    object_model.load_state_dict(checkpoint["objectness"])
    local_model.load_state_dict(checkpoint["localization"])
    records = []
    with CATLossData(args.manifest, args.fixed_grids, args.truth_cache) as data:
        for frame in args.frames:
            images = data.read_causal_stack(
                "train", frame, args.aoi, cfg["temporal_channels"]
            )
            truth = data.truth_xy("train", frame, args.aoi)
            _, regions = extract_foreground_regions(
                images, foreground_quantile=args.foreground_quantile
            )
            regions = [
                r for r in regions
                if valid_center(
                    round(r.center_x), round(r.center_y),
                    images[-1].shape[1], images[-1].shape[0], 21
                )
            ]
            object_x = np.stack([
                crop_stack(images, round(r.center_x), round(r.center_y), 21)
                for r in regions
            ])
            with torch.no_grad():
                scores = torch.softmax(
                    object_model(torch.from_numpy(object_x).to(device)), dim=1
                )[:, 1].cpu().numpy()
            accepted = [
                r for r, score in zip(regions, scores)
                if score >= args.objectness_threshold
            ]
            heatmaps = {}
            local_regions = [
                r for r in accepted
                if valid_center(
                    round(r.center_x), round(r.center_y),
                    images[-1].shape[1], images[-1].shape[0], 45
                )
            ]
            if local_regions:
                local_x = np.stack([
                    crop_stack(images, round(r.center_x), round(r.center_y), 45)
                    for r in local_regions
                ])
                with torch.no_grad():
                    values = local_model(
                        torch.from_numpy(local_x).to(device)
                    ).cpu()
                heatmaps = {id(r): values[i:i + 1] for i, r in enumerate(local_regions)}
            records.append((accepted, heatmaps, truth))

    sweep = []
    for area_threshold in args.area_thresholds:
        for peak_threshold in args.peak_thresholds:
            tp = fp = fn = detection_count = 0
            for accepted, heatmaps, truth in records:
                predictions = []
                for region in accepted:
                    heatmap = heatmaps.get(id(region))
                    if region.area < area_threshold or heatmap is None:
                        predictions.append((region.center_x, region.center_y))
                        continue
                    peaks = extract_peaks(heatmap, peak_threshold)[0]
                    if not peaks:
                        predictions.append((region.center_x, region.center_y))
                        continue
                    origin_x = round(region.center_x) - 22
                    origin_y = round(region.center_y) - 22
                    predictions.extend(
                        (origin_x + 3 * x, origin_y + 3 * y)
                        for x, y, _ in peaks
                    )
                match = match_points(predictions, truth, args.match_radius)
                tp += match.true_positives
                fp += match.false_positives
                fn += match.false_negatives
                detection_count += len(predictions)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            sweep.append({
                "complex_area_threshold": area_threshold,
                "peak_threshold": peak_threshold,
                "detections": detection_count,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
            })
    selected = max(sweep, key=lambda row: (row["f1"], row["recall"]))
    result = {
        "schema_version": 1,
        "status": "ok",
        "data_scope": "train_validation_only",
        "self_test_used": False,
        "frames": args.frames,
        "selected": selected,
        "sweep": sweep,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--aoi", default="01")
    parser.add_argument("--foreground-quantile", type=float, default=0.95)
    parser.add_argument("--objectness-threshold", type=float, default=0.58)
    parser.add_argument("--area-thresholds", type=int, nargs="+", default=[50, 100, 200, 400, 1001])
    parser.add_argument("--peak-thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 0.9])
    parser.add_argument("--match-radius", type=float, default=10.0)
    args = parser.parse_args()
    print(json.dumps(run(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
