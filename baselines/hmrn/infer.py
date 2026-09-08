"""Causal HMRN inference with previous-detection heatmap feedback and tracking."""

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

from hmrn.model import HMRN
from hmrn.real_data import HMRNData
from hmrn.targets import gaussian_heatmap
from hmrn.tracking import TrackPoint, associate, decode


def _normalize(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        return values / float(np.iinfo(image.dtype).max)
    maximum = float(np.nanmax(values)) if values.size else 0
    return values / maximum if maximum > 1 else values


def make_input(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_tracks: list[TrackPoint],
    *,
    sigma: float = 6.0,
) -> torch.Tensor:
    if previous_image.shape != current_image.shape:
        raise ValueError("Consecutive images must share a shape")
    height, width = current_image.shape
    prior_heatmap = gaussian_heatmap(
        height,
        width,
        [(track.x, track.y) for track in previous_tracks],
        sigma,
    ).numpy()
    return torch.from_numpy(
        np.stack(
            [_normalize(current_image), _normalize(previous_image), prior_heatmap]
        )
    )


def infer_frame(
    model: HMRN,
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_tracks: list[TrackPoint],
    next_track_id: int,
    device: torch.device,
    *,
    threshold: float,
    association_radius: float = 20,
) -> tuple[list[TrackPoint], int]:
    inputs = make_input(previous_image, current_image, previous_tracks)
    with torch.no_grad():
        output = model(inputs[None].to(device))
    detections = decode(
        output["center"], output["motion"], threshold=threshold
    )[0]
    return associate(
        detections,
        previous_tracks,
        next_track_id,
        radius=association_radius,
    )


def run(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["split"] != "self_test":
        raise ValueError("This final inference entry point is reserved for SELF-TEST")
    checkpoint_path = CODE_ROOT / config["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda:0")
    model = HMRN().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    frames = list(range(config["first_frame"], config["last_frame"] + 1))
    tracks: list[TrackPoint] = []
    next_track_id = 1
    rows = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with HMRNData(args.manifest, args.fixed_grids, args.truth_cache) as data:
        # Bootstrap the split boundary without reading TRAIN frame 611.
        previous_image = data.read_frame("self_test", frames[0], config["aoi"])
        for frame in frames:
            current_image = data.read_frame("self_test", frame, config["aoi"])
            tracks, next_track_id = infer_frame(
                model,
                previous_image,
                current_image,
                tracks,
                next_track_id,
                device,
                threshold=config["center_threshold"],
                association_radius=config["association_radius_px"],
            )
            rows.extend(
                {
                    "frame": frame,
                    "track_id": track.track_id,
                    "x": track.x,
                    "y": track.y,
                    "score": track.score,
                }
                for track in tracks
            )
            previous_image = current_image
    result = {
        "schema_version": 1,
        "status": "ok",
        "split": "self_test",
        "aoi": config["aoi"],
        "frames": [frames[0], frames[-1]],
        "tracks": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
