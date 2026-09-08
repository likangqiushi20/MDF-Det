"""Train HMRN with paper hyperparameters and memory-safe gradient accumulation."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from hmrn.data import build_training_sample
from hmrn.losses import CenterFocalLoss, masked_displacement_l1
from hmrn.model import HMRN
from hmrn.real_data import CachedHMRNData, HMRNData


def _inclusive_range(bounds: list[int]) -> list[int]:
    if len(bounds) != 2 or bounds[1] < bounds[0]:
        raise ValueError("Frame range must be [first, last]")
    return list(range(bounds[0], bounds[1] + 1))


def _batch_loss(
    model: HMRN,
    data,
    items: list[tuple[int, str, np.random.Generator]],
    device: torch.device,
    config: dict,
) -> tuple[torch.Tensor, dict]:
    samples = []
    for frame, aoi, rng in items:
        previous, current, previous_points, current_points = data.read_pair(
            "train", frame, aoi
        )
        samples.append(build_training_sample(
            previous, current, previous_points, current_points, rng,
            crop_width=config["crop_width"], crop_height=config["crop_height"],
        ))
    output = model(torch.stack([sample.inputs for sample in samples]).to(device))
    center_loss = CenterFocalLoss()(
        output["center"], torch.stack([sample.center for sample in samples]).to(device)
    )
    motion_loss = masked_displacement_l1(
        output["motion"],
        torch.stack([sample.displacement for sample in samples]).to(device),
        torch.stack([sample.displacement_mask for sample in samples]).to(device),
    )
    total = (
        config["center_loss_weight"] * center_loss
        + config["motion_loss_weight"] * motion_loss
    )
    return total, {
        "center": float(center_loss.detach()),
        "motion": float(motion_loss.detach()),
        "objects": sum(int(sample.center.eq(1).sum()) for sample in samples),
        "tracked_objects": sum(int(sample.displacement_mask.sum()) for sample in samples),
        "samples": len(samples),
    }


def train(
    config_path: Path,
    manifest_path: Path,
    grids_path: Path,
    truth_cache_path: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("test_set_used", False):
        raise ValueError("Training configuration must never use SELF-TEST")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda:0")
    model = HMRN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[config["lr_drop_epoch"]],
        gamma=config["lr_drop_factor"],
    )
    micro_batch_size = config["micro_batch_size"]
    accumulation = config["effective_batch_size"] // micro_batch_size
    if config["effective_batch_size"] % config["micro_batch_size"]:
        raise ValueError("Effective batch size must divide by micro batch size")
    train_frames = _inclusive_range(config["train_frames"])
    validation_frames = _inclusive_range(config["validation_frames"])
    history = []
    best_validation = float("inf")
    best_state = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    frame_cache = config.get("frame_cache")
    data_context = (
        CachedHMRNData(CODE_ROOT / frame_cache, truth_cache_path)
        if frame_cache else HMRNData(manifest_path, grids_path, truth_cache_path)
    )
    with data_context as data:
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            training = []
            ordered = rng.permutation(train_frames).tolist()
            for batch_index, start in enumerate(range(0, len(ordered), micro_batch_size), start=1):
                items = [
                    (
                        int(frame),
                        config["train_aois"][int(rng.integers(len(config["train_aois"])))],
                        rng,
                    )
                    for frame in ordered[start : start + micro_batch_size]
                ]
                loss, stats = _batch_loss(model, data, items, device, config)
                (loss / accumulation).backward()
                training.append({"total": float(loss.detach()), **stats})
                if batch_index % accumulation == 0 or start + micro_batch_size >= len(ordered):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            model.eval()
            validation = []
            with torch.no_grad():
                validation_items = []
                for frame in validation_frames:
                    for aoi in config["train_aois"]:
                        validation_rng = np.random.default_rng(
                            seed + epoch * 1_000_000 + frame * 100 + int(aoi)
                        )
                        validation_items.append((frame, aoi, validation_rng))
                for start in range(0, len(validation_items), micro_batch_size):
                        loss, stats = _batch_loss(
                            model, data, validation_items[start:start + micro_batch_size], device, config
                        )
                        validation.append({"total": float(loss), **stats})
            validation_mean = float(
                np.mean([item["total"] for item in validation])
            )
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_total": float(
                        np.mean([item["total"] for item in training])
                    ),
                    "validation_total": validation_mean,
                }
            )
            if validation_mean < best_validation:
                best_validation = validation_mean
                best_state = {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
            scheduler.step()
    assert best_state is not None
    checkpoint = CODE_ROOT / config["checkpoint"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "config": config}, checkpoint)
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "checkpoint": str(checkpoint.resolve()),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "frame_cache": str((CODE_ROOT / frame_cache).resolve()) if frame_cache else None,
        "micro_batch_size": micro_batch_size,
        "test_set_used": False,
    }
    report = CODE_ROOT / config["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
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
    args = parser.parse_args()
    print(
        json.dumps(
            train(args.config, args.manifest, args.fixed_grids, args.truth_cache)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
