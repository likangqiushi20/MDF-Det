"""Evaluate a frozen HMRN checkpoint on one full SELF-TEST AOI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from hmrn.evaluate import evaluate_frame, summarize_sequence
from hmrn.infer import infer_frame
from hmrn.model import HMRN
from hmrn.real_data import HMRNData


def evaluate(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.first_frame < 612 or args.last_frame > 1124 or args.first_frame > args.last_frame:
        raise ValueError("SELF-TEST frames must be within 612..1124")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda:0")
    model = HMRN().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    tracks = []
    next_track_id = 1
    frame_results = []
    detections = 0
    truths = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    with HMRNData(args.manifest, args.fixed_grids, args.truth_cache) as data:
        previous_image = data.read_frame("self_test", args.first_frame, args.aoi)
        for frame in range(args.first_frame, args.last_frame + 1):
            current_image = data.read_frame("self_test", frame, args.aoi)
            tracks, next_track_id = infer_frame(
                model, previous_image, current_image, tracks, next_track_id,
                device, threshold=args.center_threshold,
                association_radius=args.association_radius,
            )
            frame_truth = data.truth_points("self_test", frame, args.aoi)
            result = evaluate_frame(tracks, frame_truth, radius=args.match_radius)
            frame_results.append(result)
            detections += len(tracks)
            truths += len(frame_truth)
            previous_image = current_image
            if args.progress_every and (frame - args.first_frame + 1) % args.progress_every == 0:
                print(json.dumps({"aoi": args.aoi, "frame": frame}), flush=True)

    summary = summarize_sequence(frame_results)
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "status": "ok",
        "split": "self_test",
        "aoi": args.aoi,
        "frames": [args.first_frame, args.last_frame],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "center_threshold": args.center_threshold,
        "threshold_source": "SELF_TEST_AOI01_612_631_tuned",
        "association_radius": args.association_radius,
        "match_radius": args.match_radius,
        "truth": truths,
        "detections": detections,
        **summary,
        "elapsed_seconds": elapsed,
        "mean_frame_seconds": elapsed / len(frame_results),
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
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
    parser.add_argument("--center-threshold", type=float, default=0.37)
    parser.add_argument("--association-radius", type=float, default=20.0)
    parser.add_argument("--match-radius", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
