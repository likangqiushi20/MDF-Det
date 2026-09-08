"""Compare localization checkpoints only on crowded TRAIN validation patches."""

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
from catloss.model import LocalizationNetwork
from catloss.real_data import CATLossData
from common.metrics.point_matching import match_points


def parse_checkpoint(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--aoi", default="01")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--minimum-r", type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    samples = []
    with CATLossData(args.manifest, args.fixed_grids, args.truth_cache) as data:
        for frame in args.frames:
            images = data.read_causal_stack("train", frame, args.aoi, 4)
            truth = data.truth_xy("train", frame, args.aoi)
            for center_x_f, center_y_f in truth:
                center_x, center_y = round(center_x_f), round(center_y_f)
                if not valid_center(center_x, center_y, images[-1].shape[1], images[-1].shape[0], 45):
                    continue
                origin_x, origin_y = center_x - 22, center_y - 22
                local_truth = [
                    ((x - origin_x) / 3.0, (y - origin_y) / 3.0)
                    for x, y in truth
                    if origin_x <= x <= origin_x + 44
                    and origin_y <= y <= origin_y + 44
                ]
                if len(local_truth) >= args.minimum_r:
                    samples.append((crop_stack(images, center_x, center_y, 45), local_truth))
    inputs = torch.from_numpy(np.stack([sample[0] for sample in samples]))
    results = {}
    for item in args.checkpoint:
        name, path = parse_checkpoint(item)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = LocalizationNetwork(
            input_channels=config["temporal_channels"],
            dilation_mode=config["dilation_mode"],
        ).to(device).eval()
        model.load_state_dict(checkpoint["localization"])
        heatmaps = []
        with torch.no_grad():
            for start in range(0, len(inputs), 128):
                channels = config["temporal_channels"]
                heatmaps.append(
                    model(inputs[start:start + 128, -channels:].to(device)).cpu()
                )
        heatmaps = torch.cat(heatmaps)
        sweep = []
        for threshold in args.thresholds:
            tp = fp = fn = 0
            for heatmap, (_, truth) in zip(heatmaps, samples):
                predictions = [
                    (float(x), float(y))
                    for x, y, _ in extract_peaks(heatmap[None], threshold)[0]
                ]
                match = match_points(predictions, truth, radius=10 / 3)
                tp += match.true_positives
                fp += match.false_positives
                fn += match.false_negatives
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            sweep.append({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1})
        results[name] = {"selected": max(sweep, key=lambda row: row["f1"]), "sweep": sweep}
    output = {
        "schema_version": 1,
        "data_scope": "train_validation_crowded_patches",
        "self_test_used": False,
        "frames": args.frames,
        "aoi": args.aoi,
        "minimum_r": args.minimum_r,
        "patches": len(samples),
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
