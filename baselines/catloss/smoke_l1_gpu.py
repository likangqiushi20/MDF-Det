"""Overfit tiny real WPAFB patch batches for the CATLoss L1 gate."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from catloss.heatmaps import binary_center_map
from catloss.losses import BinaryFocalLoss, CrowdAwareThresholdedLoss
from catloss.model import LocalizationNetwork, ObjectnessNetwork
from common.data.fixed_grid_truth import project_truth_points
from common.labels.truth import TruthPolicy, iter_truth_tracks


def _read_grid(path: Path, grid: dict) -> np.ndarray:
    with rasterio.open(path) as source:
        with WarpedVRT(
            source,
            crs=grid["crs"],
            transform=Affine(*grid["transform"]),
            width=grid["width"],
            height=grid["height"],
            resampling=Resampling.bilinear,
            nodata=0,
        ) as vrt:
            return vrt.read(1)


def _moving_truth(path: Path, frame_number: int) -> list:
    policy = TruthPolicy(name="canonical_adjacent_any_motion")
    output = []
    for track in iter_truth_tracks(path):
        output.extend(
            point
            for point in policy.apply(track)
            if point.frame_number == frame_number
        )
    return output


def _crop_stack(
    images: list[np.ndarray],
    center_x: int,
    center_y: int,
    size: int,
) -> np.ndarray:
    half = size // 2
    return np.stack(
        [
            image[
                center_y - half : center_y + half + 1,
                center_x - half : center_x + half + 1,
            ]
            for image in images
        ]
    )


def _build_batches(
    images: list[np.ndarray],
    truth_xy: list[tuple[float, float]],
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = images[-1].shape
    valid_centers = [
        (round(x), round(y))
        for x, y in truth_xy
        if 24 <= x < width - 24 and 24 <= y < height - 24
    ][:sample_count]
    if len(valid_centers) < sample_count:
        raise RuntimeError("Not enough interior moving truth points for L1 smoke")

    localization_inputs = []
    localization_targets = []
    for center_x, center_y in valid_centers:
        localization_inputs.append(
            _crop_stack(images, center_x, center_y, 45).astype(np.float32) / 255.0
        )
        origin_x = center_x - 22
        origin_y = center_y - 22
        local_points = [
            ((x - origin_x) / 3.0, (y - origin_y) / 3.0)
            for x, y in truth_xy
            if origin_x <= x <= origin_x + 44 and origin_y <= y <= origin_y + 44
        ]
        localization_targets.append(
            binary_center_map(15, 15, local_points).numpy()
        )

    positive_inputs = [
        _crop_stack(images, x, y, 21).astype(np.float32) / 255.0
        for x, y in valid_centers
    ]
    negatives = []
    for y in range(24, height - 24, 37):
        for x in range(24, width - 24, 37):
            if all(math.hypot(x - tx, y - ty) > 20 for tx, ty in truth_xy):
                negatives.append(
                    _crop_stack(images, x, y, 21).astype(np.float32) / 255.0
                )
                if len(negatives) == sample_count:
                    break
        if len(negatives) == sample_count:
            break
    if len(negatives) < sample_count:
        raise RuntimeError("Not enough negative patches for L1 smoke")

    objectness_inputs = np.stack(positive_inputs + negatives)
    objectness_targets = np.asarray(
        [1] * sample_count + [0] * sample_count,
        dtype=np.int64,
    )
    return (
        objectness_inputs,
        objectness_targets,
        np.stack(localization_inputs),
        np.stack(localization_targets)[:, None],
    )


def _train_tiny_batch(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_function,
    steps: int,
    learning_rate: float,
) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    initial = None
    final = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(inputs), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss in CATLoss L1 smoke")
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
        final = float(loss.detach())
    assert initial is not None and final is not None
    return initial, final


def run(
    manifest_path: Path,
    fixed_grids_path: Path,
    output_path: Path,
    steps: int,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixed = json.loads(fixed_grids_path.read_text(encoding="utf-8"))
    grid = fixed["papers"]["catloss"]["grids"]["01"]
    dataset_root = Path(manifest["dataset_root"])
    lookup = {
        (frame["set"], frame["frame_number"]): frame for frame in manifest["frames"]
    }
    frame_numbers = [100, 101, 102, 103]
    images = [
        _read_grid(
            dataset_root / lookup[("train", frame)]["levels"]["r1"]["path"],
            grid,
        )
        for frame in frame_numbers
    ]
    truth_path = (
        dataset_root / manifest["frame_sets"]["train"]["truth_files"]["r1"]
    )
    truth = _moving_truth(truth_path, frame_numbers[-1])
    projected = project_truth_points(truth, grid)
    truth_xy = [(item.x, item.y) for item in projected]
    object_x, object_y, local_x, local_y = _build_batches(images, truth_xy, 8)

    object_model = ObjectnessNetwork().to(device)
    local_model = LocalizationNetwork(dilation_mode="half").to(device)
    started = time.perf_counter()
    object_initial, object_final = _train_tiny_batch(
        object_model,
        torch.from_numpy(object_x).to(device),
        torch.from_numpy(object_y).to(device),
        BinaryFocalLoss(alpha=0.25, gamma=2.0),
        steps,
        1e-3,
    )
    local_initial, local_final = _train_tiny_batch(
        local_model,
        torch.from_numpy(local_x).to(device),
        torch.from_numpy(local_y).to(device),
        CrowdAwareThresholdedLoss(tau=0.2, q=0.5),
        steps,
        1e-3,
    )
    elapsed = time.perf_counter() - started

    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "device": torch.cuda.get_device_name(0),
        "frames": frame_numbers,
        "paper_grid": "catloss",
        "aoi": "01",
        "objectness_batch": len(object_x),
        "localization_batch": len(local_x),
        "steps_per_model": steps,
        "objectness_loss": {"initial": object_initial, "final": object_final},
        "localization_catloss": {"initial": local_initial, "final": local_final},
        "elapsed_seconds": elapsed,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    if not object_final < object_initial or not local_final < local_initial:
        raise RuntimeError(f"Tiny-batch overfit gate failed: {result}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run(args.manifest, args.fixed_grids, args.output, args.steps)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
