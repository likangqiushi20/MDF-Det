"""Select an HMRN center threshold on TRAIN validation frames only."""

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
from hmrn.infer import infer_frame
from hmrn.model import HMRN
from hmrn.real_data import HMRNData


AOIS = ("01", "02", "03", "34", "40", "41")


def _metrics(counts: dict[str, int]) -> dict:
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    parser.add_argument("--association-radius", type=float, default=20.0)
    parser.add_argument("--match-radius", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    thresholds = sorted(set(args.thresholds))
    if not thresholds or thresholds[0] < 0 or thresholds[-1] > 1:
        raise ValueError("thresholds must be within [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda:0")
    model = HMRN().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    totals = {value: {"tp": 0, "fp": 0, "fn": 0} for value in thresholds}
    scores: list[float] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    with HMRNData(args.manifest, args.fixed_grids, args.truth_cache) as data:
        for aoi in AOIS:
            previous_image = data.read_frame("train", 560, aoi)
            tracks = []
            next_track_id = 1
            for frame in range(561, 612):
                current_image = data.read_frame("train", frame, aoi)
                tracks, next_track_id = infer_frame(
                    model, previous_image, current_image, tracks, next_track_id,
                    device, threshold=thresholds[0],
                    association_radius=args.association_radius,
                )
                scores.extend(item.score for item in tracks)
                truths = data.truth_points("train", frame, aoi)
                truth_xy = [(item.x, item.y) for item in truths]
                for threshold in thresholds:
                    selected = [item for item in tracks if item.score >= threshold]
                    matched = match_points(
                        [(item.x, item.y) for item in selected], truth_xy,
                        args.match_radius,
                    )
                    totals[threshold]["tp"] += matched.true_positives
                    totals[threshold]["fp"] += matched.false_positives
                    totals[threshold]["fn"] += matched.false_negatives
                previous_image = current_image
                if args.progress_every and (frame - 560) % args.progress_every == 0:
                    print(json.dumps({"aoi": aoi, "frame": frame}), flush=True)
            print(json.dumps({"completed_aoi": aoi}), flush=True)

    sweep = [{"threshold": value, **_metrics(totals[value])} for value in thresholds]
    selected = max(sweep, key=lambda row: row["f1"])
    result = {
        "schema_version": 1,
        "status": "ok",
        "split": "train",
        "frames": [561, 611],
        "aois": list(AOIS),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "feedback_threshold": thresholds[0],
        "association_radius": args.association_radius,
        "match_radius": args.match_radius,
        "score_quantiles": {
            str(q): float(np.quantile(scores, q))
            for q in (0, 0.25, 0.5, 0.75, 0.9, 0.99, 1)
        } if scores else {},
        "sweep": sweep,
        "selected": selected,
        "selection_source": "TRAIN_561_611_six_AOI_only",
        "self_test_used_for_selection": False,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
