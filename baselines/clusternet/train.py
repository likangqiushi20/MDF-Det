from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .model import FoveaNet, focal_dice_loss, point_map_to_gaussian
from .utils import add_external_repo


class RoobiDataset(Dataset):
    def __init__(self, bank: Path, split: str, frame_cache: Path, truth_cache: Path, external_repo: Path, *, jitter: int = 0, limit: int = 0, frame_step: int = 1) -> None:
        add_external_repo(external_repo)
        from common.data.frame_cache import FixedGridFrameCache
        self.frame_cache_path = Path(frame_cache)
        self.external_repo = Path(external_repo)
        self.frames = FixedGridFrameCache(frame_cache)
        rows = []
        for shard in sorted((bank / "shards" / split).glob("*.npz")):
            parts = shard.stem.split("_")
            frame, aoi = int(parts[1]), parts[3]
            with np.load(shard) as values:
                rows.extend((frame, aoi, int(x), int(y), bool(pos)) for x, y, pos in zip(values["x"], values["y"], values["positive"]))
        if frame_step > 1 and rows:
            first_frame = min(row[0] for row in rows)
            rows = [row for row in rows if (row[0] - first_frame) % frame_step == 0]
        if limit and len(rows) > limit:
            rng = np.random.default_rng(20260902 if split == "train" else 20260903)
            rows = [rows[int(i)] for i in rng.choice(len(rows), size=limit, replace=False)]
        self.rows = rows
        truth_file = np.load(truth_cache)
        mask = (truth_file["split"] == 0) & truth_file["moving"]
        self.truth: dict[tuple[int, str], list[tuple[float, float]]] = {}
        for frame, aoi, x, y in zip(truth_file["frame"][mask], truth_file["aoi"][mask], truth_file["x"][mask], truth_file["y"][mask]):
            key = (int(frame), f"{int(aoi):02d}")
            self.truth.setdefault(key, []).append((float(x), float(y)))
        truth_file.close()
        self.jitter = int(jitter)
        self._cached_key = None
        self._cached_stack = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Windows spawn must not pickle the opened multi-gigabyte memmaps.
        state["frames"] = None
        state["_cached_key"] = None
        state["_cached_stack"] = None
        return state

    def _ensure_frames(self) -> None:
        if self.frames is None:
            add_external_repo(self.external_repo)
            from common.data.frame_cache import FixedGridFrameCache
            self.frames = FixedGridFrameCache(self.frame_cache_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        self._ensure_frames()
        frame, aoi, x0, y0, positive = self.rows[index]
        height, width = self.frames.shape(aoi)
        points = self.truth.get((frame, aoi), [])
        if positive and self.jitter:
            old = (x0, y0)
            for _ in range(4):
                candidate_x = int(np.clip(x0 + random.randint(-self.jitter, self.jitter), 0, width - 128))
                candidate_y = int(np.clip(y0 + random.randint(-self.jitter, self.jitter), 0, height - 128))
                if any(candidate_x <= x < candidate_x + 128 and candidate_y <= y < candidate_y + 128 for x, y in points):
                    x0, y0 = candidate_x, candidate_y
                    break
            else:
                x0, y0 = old
        key = (frame, aoi)
        if key != self._cached_key:
            self._cached_stack = np.stack([
                self.frames.read_frame(value, aoi)
                for value in range(frame - 2, frame + 3)
            ])
            self._cached_key = key
        stack = self._cached_stack[:, y0:y0 + 128, x0:x0 + 128].astype(np.float32) / 255.0
        local = [(x, y) for x, y in points if x0 <= x < x0 + 128 and y0 <= y < y0 + 128]
        point_map = np.zeros((1, 64, 64), dtype=np.float32)
        for x, y in local:
            px = int(np.clip(round((x - x0) / 2.0), 0, 63))
            py = int(np.clip(round((y - y0) / 2.0), 0, 63))
            point_map[0, py, px] = 1.0
        return torch.from_numpy(stack), torch.from_numpy(point_map), torch.tensor(bool(local), dtype=torch.bool)


class FrameGroupedBatchSampler(Sampler[list[int]]):
    """Shuffle frame/AOI groups while keeping disk reads locally contiguous."""

    def __init__(self, rows, batch_size: int, shuffle: bool, seed: int = 20260902) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        groups: dict[tuple[int, str], list[int]] = {}
        for index, (frame, aoi, *_rest) in enumerate(rows):
            groups.setdefault((frame, aoi), []).append(index)
        self.groups = list(groups.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        groups = list(self.groups)
        if self.shuffle:
            rng.shuffle(groups)
        for indexes in groups:
            indexes = list(indexes)
            if self.shuffle:
                rng.shuffle(indexes)
            for start in range(0, len(indexes), self.batch_size):
                yield indexes[start:start + self.batch_size]

    def __len__(self) -> int:
        return sum((len(group) + self.batch_size - 1) // self.batch_size for group in self.groups)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def patch_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    best = None
    for threshold in np.linspace(0.05, 0.95, 37):
        predicted = scores >= threshold
        tp = int(np.count_nonzero(predicted & labels))
        fp = int(np.count_nonzero(predicted & ~labels))
        fn = int(np.count_nonzero(~predicted & labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1}
        if best is None or row["f1"] > best["f1"]:
            best = row
    return best or {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp=True) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "focal": 0.0, "dice": 0.0, "batches": 0}
    scores, labels = [], []
    for inputs, point_map, positive in loader:
        inputs = inputs.to(device, non_blocking=True)
        point_map = point_map.to(device, non_blocking=True)
        target = point_map_to_gaussian(point_map)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(inputs)
            loss, parts = focal_dice_loss(logits, target)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        totals["loss"] += float(loss.detach())
        totals["focal"] += float(parts["focal"])
        totals["dice"] += float(parts["dice"])
        totals["batches"] += 1
        if not training:
            scores.extend(torch.sigmoid(logits).flatten(1).amax(1).detach().cpu().tolist())
            labels.extend(positive.numpy().tolist())
    batches = max(1, totals.pop("batches"))
    result = {name: value / batches for name, value in totals.items()}
    if not training:
        result["patch"] = patch_metrics(np.asarray(scores), np.asarray(labels, dtype=bool))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train FoveaNet-v2 from an offline ROOBI coordinate bank.")
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--cluster-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--jitter", type=int, default=16)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--train-frame-step", type=int, default=1)
    parser.add_argument("--val-frame-step", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    # Frame-grouped batches have several tail sizes; exhaustive cuDNN tuning
    # for every distinct size costs more than it saves on Windows.
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    train_set = RoobiDataset(args.bank, "train", args.frame_cache, args.truth_cache, args.external_repo, jitter=args.jitter, limit=args.max_train_samples, frame_step=args.train_frame_step)
    val_set = RoobiDataset(args.bank, "val", args.frame_cache, args.truth_cache, args.external_repo, jitter=0, limit=args.max_val_samples, frame_step=args.val_frame_step)
    print(json.dumps({"train_samples": len(train_set), "val_samples": len(val_set), "train_frame_step": args.train_frame_step, "val_frame_step": args.val_frame_step}), flush=True)
    loader_args = dict(num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker)
    if args.workers:
        loader_args["persistent_workers"] = True
        loader_args["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_set,
        batch_sampler=FrameGroupedBatchSampler(train_set.rows, args.batch_size, True),
        **loader_args,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=FrameGroupedBatchSampler(val_set.rows, args.batch_size, False),
        **loader_args,
    )
    device = torch.device("cuda:0")
    cluster_source = torch.load(args.cluster_checkpoint, map_location="cpu", weights_only=False)
    cluster_state = cluster_source["cluster"]
    model = FoveaNet(5, base_channels=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_f1, bad_epochs, history = 1, -1.0, 0, []
    latest = args.output / "latest.pt"
    if args.resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("fovea", state.get("model")))
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_f1 = float(state["best_f1"])
        history = state.get("history", [])
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_result = run_epoch(model, train_loader, device, optimizer, scaler)
        with torch.inference_mode():
            val_result = run_epoch(model, val_loader, device)
        val_f1 = val_result["patch"]["f1"]
        scheduler.step(val_f1)
        row = {"epoch": epoch, "train": train_result, "val": val_result, "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started}
        history.append(row)
        state = {"epoch": epoch, "cluster": cluster_state, "fovea": model.state_dict(), "optimizer": optimizer.state_dict(), "best_f1": max(best_f1, val_f1), "history": history, "args": vars(args), "model_config": {"temporal_channels": 5, "base_channels": 16}}
        torch.save(state, latest)
        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            torch.save(state, args.output / "best.pt")
        else:
            bad_epochs += 1
        (args.output / "history.json").write_text(json.dumps(history, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)
        if bad_epochs >= args.patience:
            print(json.dumps({"early_stopping": True, "best_patch_f1": best_f1}), flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
