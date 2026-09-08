"""Train CATLoss objectness and localization networks on fixed-grid WAMI patches."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from catloss.data import (
    PatchPools,
    augment_patch_batches,
    build_localization_batch,
    build_patch_pools,
    sample_objectness_epoch,
)
from catloss.losses import (
    BinaryFocalLoss,
    CrowdAwareThresholdedLoss,
    ThresholdedLoss,
)
from catloss.model import LocalizationNetwork, ObjectnessNetwork
from catloss.real_data import CATLossData, CachedCATLossData


def _configured_frames(config: dict, name: str) -> list[int]:
    range_name = name.replace("_frames", "_frame_range")
    if range_name in config:
        first, last = config[range_name]
        return list(range(first, last + 1))
    return config[name]


def make_localization_loss(config: dict) -> torch.nn.Module:
    name = config.get("localization_loss", "catloss")
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "tloss":
        return ThresholdedLoss(tau=config["tau"])
    if name == "catloss":
        return CrowdAwareThresholdedLoss(tau=config["tau"], q=config["q"])
    raise ValueError(f"Unsupported localization_loss: {name}")


def _limited_pools(
    pools: PatchPools,
    maximum: int,
    rng: np.random.Generator,
) -> PatchPools:
    if len(pools.positive) <= maximum:
        return pools
    indices = rng.choice(len(pools.positive), maximum, replace=False)
    return replace(
        pools,
        positive=tuple(pools.positive[index] for index in indices),
    )


def _frame_loss(
    data: CATLossData,
    frame: int,
    config: dict,
    rng: np.random.Generator,
    object_model: ObjectnessNetwork,
    local_model: LocalizationNetwork,
    object_loss_function: BinaryFocalLoss,
    local_loss_function: CrowdAwareThresholdedLoss,
    device: torch.device,
    aoi: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    selected_aoi = aoi or config["aoi"]
    images = data.read_causal_stack(
        "train",
        frame,
        selected_aoi,
        channels=config.get("bgs_temporal_channels", 4),
    )
    model_images = images[-config["temporal_channels"] :]
    truth_xy = data.truth_xy("train", frame, selected_aoi)
    pools = _limited_pools(
        build_patch_pools(images, truth_xy),
        config["max_positive_patches_per_frame"],
        rng,
    )
    object_x, object_y = sample_objectness_epoch(
        model_images,
        pools,
        rng,
        negatives_per_positive=config["negatives_per_positive"],
        hard_fraction=config["hard_negative_fraction"],
    )
    localization_centers = [
        center
        for center in pools.positive
        if 22 <= center[0] < images[-1].shape[1] - 22
        and 22 <= center[1] < images[-1].shape[0] - 22
    ]
    local_x, local_y = build_localization_batch(
        model_images,
        truth_xy,
        localization_centers,
    )
    if config.get("augmentation", False):
        object_x, local_x, local_y = augment_patch_batches(
            object_x, local_x, local_y, rng
        )
    object_loss = object_loss_function(
        object_model(torch.from_numpy(object_x).to(device)),
        torch.from_numpy(object_y).to(device),
    )
    local_loss = local_loss_function(
        local_model(torch.from_numpy(local_x).to(device)),
        torch.from_numpy(local_y).to(device),
    )
    stats = {
        "frame": frame,
        "aoi": selected_aoi,
        "truth": len(truth_xy),
        "positive_patches": len(pools.positive),
        "hard_negative_pool": len(pools.hard_negative),
        "objectness_batch": len(object_x),
        "localization_batch": len(local_x),
    }
    return object_loss, local_loss, stats


def train(
    config_path: Path,
    manifest_path: Path,
    fixed_grids_path: Path,
    truth_cache_path: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("test_set_used", False):
        raise ValueError("CATLoss training must never use SELF-TEST")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CATLoss training")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda:0")

    object_model = ObjectnessNetwork(
        input_channels=config["temporal_channels"]
    ).to(device)
    local_model = LocalizationNetwork(
        input_channels=config["temporal_channels"],
        dilation_mode=config["dilation_mode"],
    ).to(device)
    object_optimizer = torch.optim.AdamW(
        object_model.parameters(),
        lr=config["objectness_learning_rate"],
        weight_decay=config["weight_decay"],
    )
    local_optimizer = torch.optim.AdamW(
        local_model.parameters(),
        lr=config["localization_learning_rate"],
        weight_decay=config["weight_decay"],
    )
    object_loss_function = BinaryFocalLoss()
    localization_loss_name = config.get("localization_loss", "catloss")
    local_loss_function = make_localization_loss(config)

    history = []
    best_validation = float("inf")
    best_states = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    frame_cache = config.get("frame_cache")
    data_context = (
        CachedCATLossData(CODE_ROOT / frame_cache, truth_cache_path)
        if frame_cache else CATLossData(manifest_path, fixed_grids_path, truth_cache_path)
    )
    with data_context as data:
        for epoch in range(config["epochs"]):
            object_model.train()
            local_model.train()
            training_losses = []
            frame_stats = []
            frame_order = rng.permutation(
                _configured_frames(config, "train_frames")
            ).tolist()
            train_aois = config.get("train_aois", [config["aoi"]])
            for frame in frame_order:
                selected_aoi = train_aois[int(rng.integers(len(train_aois)))]
                object_optimizer.zero_grad(set_to_none=True)
                local_optimizer.zero_grad(set_to_none=True)
                object_loss, local_loss, stats = _frame_loss(
                    data,
                    frame,
                    config,
                    rng,
                    object_model,
                    local_model,
                    object_loss_function,
                    local_loss_function,
                    device,
                    selected_aoi,
                )
                object_loss.backward()
                local_loss.backward()
                object_optimizer.step()
                local_optimizer.step()
                training_losses.append(
                    [float(object_loss.detach()), float(local_loss.detach())]
                )
                frame_stats.append(stats)

            object_model.eval()
            local_model.eval()
            validation_losses = []
            with torch.no_grad():
                validation_aois = config.get(
                    "validation_aois", [config["aoi"]]
                )
                for frame in _configured_frames(config, "validation_frames"):
                    for selected_aoi in validation_aois:
                        object_loss, local_loss, _ = _frame_loss(
                        data,
                        frame,
                        config,
                        rng,
                        object_model,
                        local_model,
                        object_loss_function,
                        local_loss_function,
                        device,
                        selected_aoi,
                        )
                        validation_losses.append(
                            [float(object_loss), float(local_loss)]
                        )
            train_mean = np.mean(training_losses, axis=0).tolist()
            validation_mean = np.mean(validation_losses, axis=0).tolist()
            validation_total = sum(validation_mean)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_objectness": train_mean[0],
                    "train_localization": train_mean[1],
                    "validation_objectness": validation_mean[0],
                    "validation_localization": validation_mean[1],
                    "frame_stats": frame_stats,
                }
            )
            if validation_total < best_validation:
                best_validation = validation_total
                best_states = {
                    "objectness": {
                        key: value.detach().cpu()
                        for key, value in object_model.state_dict().items()
                    },
                    "localization": {
                        key: value.detach().cpu()
                        for key, value in local_model.state_dict().items()
                    },
                    "epoch": epoch + 1,
                }

    assert best_states is not None
    checkpoint_path = CODE_ROOT / config["checkpoint"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **best_states,
            "config": config,
            "paper_spec": json.loads(
                (CODE_ROOT / "catloss" / "paper_spec.json").read_text(
                    encoding="utf-8"
                )
            ),
        },
        checkpoint_path,
    )
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "device": torch.cuda.get_device_name(0),
        "config": str(config_path.resolve()),
        "localization_loss": localization_loss_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": best_states["epoch"],
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "frame_cache": str((CODE_ROOT / frame_cache).resolve()) if frame_cache else None,
        "test_set_used": False,
    }
    report_path = CODE_ROOT / config["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
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
            train(
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
