"""Select an objectness threshold using training-set validation frames only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from catloss.data import crop_stack, valid_center
from catloss.model import ObjectnessNetwork
from catloss.postprocess import extract_foreground_regions
from catloss.real_data import CATLossData
from common.metrics.point_matching import match_points


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = torch.load(
        CODE_ROOT / args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    train_config = checkpoint["config"]
    device = torch.device("cuda:0")
    model = ObjectnessNetwork(
        input_channels=train_config["temporal_channels"]
    ).to(device)
    model.load_state_dict(checkpoint["objectness"])
    model.eval()

    frame_records = []
    with CATLossData(args.manifest, args.fixed_grids, args.truth_cache) as data:
        for frame in args.frames:
            images = data.read_causal_stack(
                "train",
                frame,
                args.aoi,
                channels=train_config.get("bgs_temporal_channels", 4),
            )
            model_images = images[-train_config["temporal_channels"] :]
            truth_xy = data.truth_xy("train", frame, args.aoi)
            _, regions = extract_foreground_regions(
                images,
                foreground_quantile=args.foreground_quantile,
            )
            regions = [
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
            inputs = np.stack(
                [
                    crop_stack(
                        model_images,
                        round(region.center_x),
                        round(region.center_y),
                        21,
                    )
                    for region in regions
                ]
            )
            with torch.no_grad():
                scores = torch.softmax(
                    model(torch.from_numpy(inputs).to(device)),
                    dim=1,
                )[:, 1].cpu().numpy()
            frame_records.append((regions, scores, truth_xy))

    sweep = []
    for threshold in np.linspace(args.threshold_min, args.threshold_max, args.steps):
        tp = fp = fn = 0
        for regions, scores, truth_xy in frame_records:
            predictions = [
                (region.center_x, region.center_y)
                for region, score in zip(regions, scores)
                if score >= threshold
            ]
            matches = match_points(predictions, truth_xy, args.match_radius)
            tp += matches.true_positives
            fp += matches.false_positives
            fn += matches.false_negatives
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        sweep.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    selected = max(sweep, key=lambda item: (item["f1"], item["recall"]))
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "data_scope": "train_validation_only",
        "self_test_used": False,
        "frames": args.frames,
        "aoi": args.aoi,
        "foreground_quantile": args.foreground_quantile,
        "selected": selected,
        "sweep": sweep,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--aoi", default="01")
    parser.add_argument("--foreground-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-min", type=float, default=0.35)
    parser.add_argument("--threshold-max", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=31)
    parser.add_argument("--match-radius", type=float, default=10.0)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
