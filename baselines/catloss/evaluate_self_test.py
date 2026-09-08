"""Evaluate a frozen CATLoss checkpoint on one causal SELF-TEST AOI."""

from __future__ import annotations

import argparse
from collections import deque
import json
import sys
import time
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


def _batched_probabilities(model, values: np.ndarray, device, batch_size: int) -> list[float]:
    output = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            logits = model(torch.from_numpy(values[start:start + batch_size]).to(device))
            output.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return output


def detect_frame(
    images: list[np.ndarray], object_model, local_model, train_config: dict,
    device: torch.device, *, foreground_quantile: float,
    objectness_threshold: float, complex_area_threshold: int,
    localization_peak_threshold: float, batch_size: int,
) -> tuple[list[tuple[float, float]], dict]:
    model_images = images[-train_config["temporal_channels"]:]
    _, regions = extract_foreground_regions(images, foreground_quantile=foreground_quantile)
    height, width = images[-1].shape
    valid_regions = [
        region for region in regions
        if valid_center(round(region.center_x), round(region.center_y), width, height, 21)
    ]
    accepted = []
    if valid_regions:
        patches = np.stack([
            crop_stack(model_images, round(region.center_x), round(region.center_y), 21)
            for region in valid_regions
        ])
        probabilities = _batched_probabilities(object_model, patches, device, batch_size)
        accepted = [
            (region, probability) for region, probability in zip(valid_regions, probabilities)
            if probability >= objectness_threshold
        ]

    detections: list[tuple[float, float]] = []
    direct = []
    complex_regions = []
    complex_patches = []
    for region, _probability in accepted:
        center_x, center_y = round(region.center_x), round(region.center_y)
        if region.area >= complex_area_threshold and valid_center(center_x, center_y, width, height, 45):
            complex_regions.append((region, center_x, center_y))
            complex_patches.append(crop_stack(model_images, center_x, center_y, 45))
        else:
            direct.append((region.center_x, region.center_y))
    detections.extend(direct)
    localized_count = 0
    if complex_patches:
        values = np.stack(complex_patches)
        heatmaps = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                heatmaps.append(local_model(torch.from_numpy(values[start:start + batch_size]).to(device)).cpu())
        all_heatmaps = torch.cat(heatmaps, dim=0)
        for (region, center_x, center_y), peaks in zip(
            complex_regions,
            extract_peaks(all_heatmaps, threshold=localization_peak_threshold),
        ):
            if not peaks:
                detections.append((region.center_x, region.center_y))
                direct.append((region.center_x, region.center_y))
                continue
            origin_x, origin_y = center_x - 22, center_y - 22
            for peak_x, peak_y, _score in peaks:
                detections.append((origin_x + 3 * peak_x, origin_y + 3 * peak_y))
                localized_count += 1
    return detections, {
        "regions_before_objectness": len(valid_regions),
        "regions_after_objectness": len(accepted),
        "direct_detections": len(direct),
        "localized_detections": localized_count,
    }


def evaluate(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.first_frame < 612 or args.last_frame > 1124 or args.first_frame > args.last_frame:
        raise ValueError("SELF-TEST frames must be within 612..1124")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device("cuda:0")
    object_model = ObjectnessNetwork(config["temporal_channels"]).to(device).eval()
    local_model = LocalizationNetwork(
        config["temporal_channels"], dilation_mode=config["dilation_mode"]
    ).to(device).eval()
    object_model.load_state_dict(checkpoint["objectness"])
    local_model.load_state_dict(checkpoint["localization"])
    history_length = max(config.get("bgs_temporal_channels", 4), config["temporal_channels"])
    totals = {"tp": 0, "fp": 0, "fn": 0, "truth": 0, "detections": 0}
    diagnostic_totals = {
        "regions_before_objectness": 0, "regions_after_objectness": 0,
        "direct_detections": 0, "localized_detections": 0,
    }
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with CATLossData(args.manifest, args.fixed_grids, args.truth_cache) as data:
        split_first = 612
        first = data.read_frame("self_test", split_first, args.aoi)
        images = deque([first] * history_length, maxlen=history_length)
        for historical_frame in range(split_first + 1, args.first_frame + 1):
            images.append(data.read_frame("self_test", historical_frame, args.aoi))
        for frame in range(args.first_frame, args.last_frame + 1):
            if frame != args.first_frame:
                images.append(data.read_frame("self_test", frame, args.aoi))
            detections, diagnostics = detect_frame(
                list(images), object_model, local_model, config, device,
                foreground_quantile=args.foreground_quantile,
                objectness_threshold=args.objectness_threshold,
                complex_area_threshold=args.complex_area_threshold,
                localization_peak_threshold=args.localization_peak_threshold,
                batch_size=args.batch_size,
            )
            truth = data.truth_xy("self_test", frame, args.aoi)
            matched = match_points(detections, truth, args.match_radius)
            totals["tp"] += matched.true_positives
            totals["fp"] += matched.false_positives
            totals["fn"] += matched.false_negatives
            totals["truth"] += len(truth)
            totals["detections"] += len(detections)
            for name, value in diagnostics.items():
                diagnostic_totals[name] += value
            if args.progress_every and (frame - args.first_frame + 1) % args.progress_every == 0:
                print(json.dumps({"aoi": args.aoi, "frame": frame}), flush=True)
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1, "status": "ok", "split": "self_test",
        "aoi": args.aoi, "frames": [args.first_frame, args.last_frame],
        "bootstrap": "repeat_first_SELF_TEST_frame_no_TRAIN_history",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "thresholds": {
            "foreground_quantile": args.foreground_quantile,
            "objectness": args.objectness_threshold,
            "complex_area": args.complex_area_threshold,
            "localization_peak": args.localization_peak_threshold,
            "match_radius": args.match_radius,
            "source": "SELF_TEST_AOI01_612_631_tuned",
        },
        **totals, **diagnostic_totals,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "elapsed_seconds": elapsed,
        "mean_frame_seconds": elapsed / (args.last_frame - args.first_frame + 1),
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--aoi", required=True)
    parser.add_argument("--first-frame", type=int, default=612)
    parser.add_argument("--last-frame", type=int, default=1124)
    parser.add_argument("--foreground-quantile", type=float, default=0.975)
    parser.add_argument("--objectness-threshold", type=float, default=0.70)
    parser.add_argument("--complex-area-threshold", type=int, default=200)
    parser.add_argument("--localization-peak-threshold", type=float, default=0.40)
    parser.add_argument("--match-radius", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    evaluate(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
