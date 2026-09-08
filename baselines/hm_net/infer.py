"""Causal HM-Net inference with SGR feedback and one-frame passive tracks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from hm_net.data import HMNetData
from hm_net.model import HMNet
from hm_net.sgr import selective_gaussian_reconstruction
from hm_net.tracking import Track, associate_hungarian, decode


def _image_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    values = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        values /= float(np.iinfo(image.dtype).max)
    return torch.from_numpy(values)[None, None].to(device)


def _pad_inputs(
    current: torch.Tensor,
    previous: torch.Tensor,
    feedback: torch.Tensor,
    multiple: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = current.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    padding = (0, pad_width, 0, pad_height)
    return (
        functional.pad(current, padding),
        functional.pad(previous, padding),
        functional.pad(feedback, padding),
    )


def run(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["split"] != "self_test":
        raise ValueError("Final inference config must use SELF-TEST")
    device = torch.device("cuda:0")
    checkpoint = torch.load(
        CODE_ROOT / config["checkpoint"], map_location="cpu", weights_only=False
    )
    model = HMNet().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    frames = list(range(config["first_frame"], config["last_frame"] + 1))
    active: list[Track] = []
    passive: list[Track] = []
    next_id = 1
    rows = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with HMNetData(args.manifest, args.fixed_grids, args.truth_cache) as data:
        previous_image = data.read_frame("self_test", frames[0], config["aoi"])
        height, width = previous_image.shape
        feedback = torch.zeros(1, 2, height, width, device=device)
        for frame in frames:
            current_image = data.read_frame("self_test", frame, config["aoi"])
            current_tensor, previous_tensor, padded_feedback = _pad_inputs(
                _image_tensor(current_image, device),
                _image_tensor(previous_image, device),
                feedback,
            )
            with torch.no_grad():
                prediction = model(
                    current_tensor,
                    previous_tensor,
                    padded_feedback,
                )
                prediction = {
                    name: values[..., :height, :width]
                    for name, values in prediction.items()
                }
            detections = decode(
                prediction["center"],
                prediction["motion"],
                prediction["precision"],
                threshold=config["detection_threshold"],
            )[0]
            candidates = active + passive
            tracks, next_id = associate_hungarian(
                detections,
                candidates,
                next_id,
                gate=config["association_gate_px"],
            )
            matched_ids = {track.track_id for track in tracks}
            passive = [
                track for track in active if track.track_id not in matched_ids
            ]
            active = tracks
            feedback = selective_gaussian_reconstruction(
                prediction["center"],
                filter_threshold=config["sgr_threshold"],
                detection_threshold=config["detection_threshold"],
                amplification=config["sgr_amplification"],
            )
            rows.extend(
                {
                    "frame": frame,
                    "track_id": track.track_id,
                    "x": track.x,
                    "y": track.y,
                    "class_id": track.class_id,
                    "score": track.score,
                }
                for track in tracks
                if track.score >= config["detection_threshold"]
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
