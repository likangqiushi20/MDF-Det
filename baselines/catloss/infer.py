"""Run the CATLoss two-stage detection pipeline on one fixed-grid frame."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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


def run(
    config_path: Path,
    manifest_path: Path,
    fixed_grids_path: Path,
    truth_cache_path: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CATLoss inference")
    device = torch.device("cuda:0")
    checkpoint_path = CODE_ROOT / config["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config = checkpoint["config"]
    object_model = ObjectnessNetwork(
        input_channels=train_config["temporal_channels"]
    ).to(device)
    local_model = LocalizationNetwork(
        input_channels=train_config["temporal_channels"],
        dilation_mode=train_config["dilation_mode"],
    ).to(device)
    object_model.load_state_dict(checkpoint["objectness"])
    local_model.load_state_dict(checkpoint["localization"])
    object_model.eval()
    local_model.eval()

    started = time.perf_counter()
    with CATLossData(manifest_path, fixed_grids_path, truth_cache_path) as data:
        images = data.read_causal_stack(
            config["split"],
            config["frame"],
            config["aoi"],
            channels=train_config.get("bgs_temporal_channels", 4),
        )
        model_images = images[-train_config["temporal_channels"] :]
        truth_xy = data.truth_xy(
            config["split"], config["frame"], config["aoi"]
        )
    _, regions = extract_foreground_regions(
        images,
        foreground_quantile=config["foreground_quantile"],
    )
    valid_regions = [
        region
        for region in regions
        if valid_center(
            round(region.center_x),
            round(region.center_y),
            images[-1].shape[1],
            images[-1].shape[0],
            21,
        )
    ]
    detections: list[tuple[float, float]] = []
    accepted = []
    probability_values: list[float] = []
    if valid_regions:
        object_inputs = np.stack(
            [
                crop_stack(
                    model_images,
                    round(region.center_x),
                    round(region.center_y),
                    21,
                )
                for region in valid_regions
            ]
        )
        with torch.no_grad():
            probabilities = torch.softmax(
                object_model(torch.from_numpy(object_inputs).to(device)),
                dim=1,
            )[:, 1].cpu()
        probability_values = probabilities.tolist()
        accepted = [
            (region, float(probability))
            for region, probability in zip(valid_regions, probabilities)
            if probability >= config["objectness_threshold"]
        ]

    direct_count = 0
    localized_count = 0
    for region, _probability in accepted:
        center_x = round(region.center_x)
        center_y = round(region.center_y)
        use_localization = (
            region.area >= config["complex_area_threshold"]
            and valid_center(
                center_x,
                center_y,
                images[-1].shape[1],
                images[-1].shape[0],
                45,
            )
        )
        if not use_localization:
            detections.append((region.center_x, region.center_y))
            direct_count += 1
            continue
        local_input = crop_stack(model_images, center_x, center_y, 45)[None]
        with torch.no_grad():
            heatmap = local_model(torch.from_numpy(local_input).to(device))
        peaks = extract_peaks(
            heatmap,
            threshold=config["localization_peak_threshold"],
        )[0]
        if not peaks:
            detections.append((region.center_x, region.center_y))
            direct_count += 1
            continue
        origin_x = center_x - 22
        origin_y = center_y - 22
        for peak_x, peak_y, _score in peaks:
            detections.append(
                (origin_x + 3 * peak_x, origin_y + 3 * peak_y)
            )
            localized_count += 1

    raw_centers = [(region.center_x, region.center_y) for region in valid_regions]
    raw_matches = match_points(
        raw_centers,
        truth_xy,
        radius=config["match_radius_px"],
    )
    matches = match_points(
        detections,
        truth_xy,
        radius=config["match_radius_px"],
    )
    precision = (
        matches.true_positives
        / (matches.true_positives + matches.false_positives)
        if detections
        else 0.0
    )
    recall = (
        matches.true_positives
        / (matches.true_positives + matches.false_negatives)
        if truth_xy
        else 0.0
    )
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "split": config["split"],
        "aoi": config["aoi"],
        "frame": config["frame"],
        "regions_before_objectness": len(valid_regions),
        "regions_after_objectness": len(accepted),
        "candidate_recall_before_objectness": (
            raw_matches.true_positives / len(truth_xy) if truth_xy else 0.0
        ),
        "objectness_probability": {
            "minimum": min(probability_values) if probability_values else None,
            "median": (
                float(np.median(probability_values))
                if probability_values
                else None
            ),
            "maximum": max(probability_values) if probability_values else None,
        },
        "direct_detections": direct_count,
        "localized_detections": localized_count,
        "detections": len(detections),
        "truth": len(truth_xy),
        "metrics": {
            "radius_px": config["match_radius_px"],
            "tp": matches.true_positives,
            "fp": matches.false_positives,
            "fn": matches.false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "threshold_status": "smoke_only_not_frozen",
    }
    output_path = CODE_ROOT / config["output"]
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                args.config,
                args.manifest,
                args.fixed_grids,
                args.truth_cache,
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
