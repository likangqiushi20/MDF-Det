"""Train HM-Net on TRAIN only using paper hyperparameters."""

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

from hm_net.data import CachedHMNetData, HMNetData, build_training_sample
from hm_net.losses import hmnet_loss
from hm_net.model import HMNet
from hm_net.tracking import decode


def _frames(bounds: list[int]) -> list[int]:
    return list(range(bounds[0], bounds[1] + 1))


def _loss_for_batch(
    model: HMNet,
    data,
    items,
    device: torch.device,
    weights: dict,
    *,
    feedback_frame_dropout_probability: float = 0.0,
    feedback_center_dropout_probability: float = 0.0,
) -> tuple[torch.Tensor, dict, dict, list]:
    samples = [
        build_training_sample(
            *data.read_pair("train", frame, aoi),
            rng,
            feedback_frame_dropout_probability=(
                feedback_frame_dropout_probability
            ),
            feedback_center_dropout_probability=(
                feedback_center_dropout_probability
            ),
        )
        for frame, aoi, rng in items
    ]
    prediction = model(
        torch.stack([sample.current for sample in samples]).to(device),
        torch.stack([sample.previous for sample in samples]).to(device),
        torch.stack([sample.feedback for sample in samples]).to(device),
    )
    total, parts = hmnet_loss(
        prediction,
        torch.stack([sample.center for sample in samples]).to(device),
        torch.stack([sample.motion for sample in samples]).to(device),
        torch.stack([sample.precision for sample in samples]).to(device),
        torch.stack([sample.motion_mask for sample in samples]).to(device),
        torch.stack([sample.precision_mask for sample in samples]).to(device),
        center_weight=weights["center"],
        motion_weight=weights["motion"],
        precision_weight=weights["precision"],
    )
    return (
        total,
        {name: float(value.detach()) for name, value in parts.items()},
        prediction,
        samples,
    )


def _greedy_point_counts(
    detections,
    truths: np.ndarray,
    radius: float,
) -> tuple[int, int, int]:
    """Fast confidence-ordered proxy used only for checkpoint selection."""
    if len(truths) == 0:
        return 0, len(detections), 0
    available = np.ones(len(truths), dtype=bool)
    true_positives = 0
    for detection in detections:
        indexes = np.flatnonzero(available)
        if not len(indexes):
            break
        distances = np.hypot(
            truths[indexes, 0] - detection.x,
            truths[indexes, 1] - detection.y,
        )
        nearest = int(np.argmin(distances))
        if distances[nearest] <= radius:
            available[indexes[nearest]] = False
            true_positives += 1
    return (
        true_positives,
        len(detections) - true_positives,
        len(truths) - true_positives,
    )


def _update_detection_validation(
    totals: dict[float, dict[str, int]],
    prediction: dict[str, torch.Tensor],
    samples: list,
    thresholds: list[float],
    radius: float,
    max_detections: int,
) -> None:
    decoded = decode(
        prediction["center"],
        prediction["motion"],
        prediction["precision"],
        threshold=min(thresholds),
        max_detections_per_class=max_detections,
    )
    for detections, sample in zip(decoded, samples):
        moving = [item for item in detections if item.class_id == 0]
        rows, columns = torch.nonzero(
            sample.center[0].eq(1), as_tuple=True
        )
        truths = np.column_stack(
            (columns.cpu().numpy(), rows.cpu().numpy())
        )
        for threshold in thresholds:
            selected = [item for item in moving if item.score >= threshold]
            tp, fp, fn = _greedy_point_counts(selected, truths, radius)
            totals[threshold]["tp"] += tp
            totals[threshold]["fp"] += fp
            totals[threshold]["fn"] += fn


def _summarize_detection_validation(totals: dict) -> dict:
    rows = []
    for threshold, counts in totals.items():
        precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append(
            {
                "threshold": threshold,
                **counts,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return max(rows, key=lambda row: row["f1"])


def train(
    config_path: Path,
    manifest_path: Path,
    grids_path: Path,
    truth_path: Path,
    *,
    resume: bool = False,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("test_set_used", False):
        raise ValueError("Training must not use SELF-TEST")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    micro_batch_size = config["micro_batch_size"]
    if config["effective_batch_size"] % micro_batch_size:
        raise ValueError("Effective batch size must divide by micro batch size")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda:0")
    model = HMNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    accumulation = config["effective_batch_size"] // micro_batch_size
    train_frames = _frames(config["train_frames"])
    validation_frames = _frames(config["validation_frames"])
    selection_metric = config.get("selection_metric", "validation_loss")
    best_value = -float("inf") if selection_metric == "detection_f1" else float("inf")
    history = []
    start_epoch = 1
    previous_elapsed = 0.0
    checkpoint = CODE_ROOT / config["checkpoint"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    latest = checkpoint.with_name(
        checkpoint.stem.replace("_best", "") + "_latest.pt"
    )
    report = CODE_ROOT / config["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    if resume and latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        best_value = saved["best_value"]
        start_epoch = saved["epoch"] + 1
        previous_elapsed = saved.get("elapsed_seconds", 0.0)
        rng.bit_generator.state = saved["numpy_rng_state"]
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    frame_cache = config.get("frame_cache")
    data_context = (
        CachedHMNetData(CODE_ROOT / frame_cache, truth_path)
        if frame_cache else HMNetData(manifest_path, grids_path, truth_path)
    )
    with data_context as data:
        for epoch in range(start_epoch, config["epochs"] + 1):
            feedback_config = config.get("feedback_robustness", {})
            schedule_fraction = (
                (epoch - 1) / max(1, config["epochs"] - 1)
            )
            frame_dropout = (
                feedback_config.get("frame_dropout_start", 0.0)
                + schedule_fraction
                * (
                    feedback_config.get("frame_dropout_end", 0.0)
                    - feedback_config.get("frame_dropout_start", 0.0)
                )
            )
            center_dropout = feedback_config.get("center_dropout", 0.0)
            validation_frame_dropout = feedback_config.get(
                "validation_frame_dropout", 0.50
            )
            validation_center_dropout = feedback_config.get(
                "validation_center_dropout", center_dropout
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_losses = []
            ordered = rng.permutation(train_frames).tolist()
            for batch_index, start in enumerate(range(0, len(ordered), micro_batch_size), start=1):
                items = [
                    (int(frame), config["train_aois"][int(rng.integers(len(config["train_aois"])))], rng)
                    for frame in ordered[start:start + micro_batch_size]
                ]
                loss, _, _, _ = _loss_for_batch(
                    model,
                    data,
                    items,
                    device,
                    config["loss_weights"],
                    feedback_frame_dropout_probability=frame_dropout,
                    feedback_center_dropout_probability=center_dropout,
                )
                (loss / accumulation).backward()
                train_losses.append(float(loss.detach()))
                if batch_index % accumulation == 0 or start + micro_batch_size >= len(ordered):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            model.eval()
            validation_losses = []
            validation_thresholds = config.get(
                "validation_detection_thresholds",
                [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            )
            validation_totals = {
                float(threshold): {"tp": 0, "fp": 0, "fn": 0}
                for threshold in validation_thresholds
            }
            with torch.no_grad():
                for frame in validation_frames:
                    # Deterministic AOI rotation and crop, with no SELF-TEST.
                    aoi = config["train_aois"][frame % len(config["train_aois"])]
                    validation_rng = np.random.default_rng(seed + frame)
                    loss, _, prediction, samples = _loss_for_batch(
                        model, data, [(frame, aoi, validation_rng)], device,
                        config["loss_weights"],
                        feedback_frame_dropout_probability=(
                            validation_frame_dropout
                        ),
                        feedback_center_dropout_probability=(
                            validation_center_dropout
                        ),
                    )
                    validation_losses.append(float(loss))
                    _update_detection_validation(
                        validation_totals,
                        prediction,
                        samples,
                        validation_thresholds,
                        config.get("validation_match_radius", 10.0),
                        config.get("validation_max_detections", 2000),
                    )
            validation = float(np.mean(validation_losses))
            detection_validation = _summarize_detection_validation(
                validation_totals
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_total": float(np.mean(train_losses)),
                    "validation_total": validation,
                    "validation_detection": detection_validation,
                    "feedback_frame_dropout": frame_dropout,
                }
            )
            selection_value = (
                detection_validation["f1"]
                if selection_metric == "detection_f1"
                else validation
            )
            improved = (
                selection_value > best_value
                if selection_metric == "detection_f1"
                else selection_value < best_value
            )
            if improved:
                best_value = selection_value
                torch.save(
                    {
                        "model": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "config": config,
                        "epoch": epoch,
                        "validation_total": validation,
                        "validation_detection": detection_validation,
                        "selection_metric": selection_metric,
                        "selection_value": selection_value,
                    },
                    checkpoint,
                )
            elapsed = previous_elapsed + time.perf_counter() - started
            torch.save(
                {
                    "model": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "history": history,
                    "best_value": best_value,
                    "elapsed_seconds": elapsed,
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                },
                latest,
            )
            progress = {
                "schema_version": 1,
                "status": "running",
                "epoch": epoch,
                "epochs": config["epochs"],
                "latest": str(latest.resolve()),
                "best_checkpoint": str(checkpoint.resolve()),
                "best_selection_value": best_value,
                "selection_metric": selection_metric,
                "history": history,
                "elapsed_seconds": elapsed,
                "test_set_used": False,
            }
            report.write_text(
                json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(progress, ensure_ascii=False), flush=True)
    if not checkpoint.is_file():
        raise RuntimeError("No best checkpoint was produced")
    total_elapsed = previous_elapsed + time.perf_counter() - started
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "checkpoint": str(checkpoint.resolve()),
        "latest_checkpoint": str(latest.resolve()),
        "history": history,
        "elapsed_seconds": total_elapsed,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "frame_cache": str((CODE_ROOT / frame_cache).resolve()) if frame_cache else None,
        "micro_batch_size": micro_batch_size,
        "selection_metric": selection_metric,
        "best_selection_value": best_value,
        "test_set_used": False,
    }
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.config,
                args.manifest,
                args.fixed_grids,
                args.truth_cache,
                resume=args.resume,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
