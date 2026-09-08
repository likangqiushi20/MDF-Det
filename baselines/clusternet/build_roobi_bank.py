from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .utils import (
    AOIS, add_external_repo, origins_cover_bounds,
)


def patch_contains_truth(origin: tuple[int, int], truth: list[tuple[float, float]], size: int = 128) -> bool:
    x0, y0 = origin
    return any(x0 <= x < x0 + size and y0 <= y < y0 + size for x, y in truth)


def build_split(
    *, split_name: str, frames: range, aois: list[str], output: Path,
    data, cluster, device, cluster_quantile: float, negative_ratio: float,
    batch_seed: int,
) -> dict:
    shard_dir = output / "shards" / split_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed = samples = positives = negatives = 0
    started = time.perf_counter()
    rng = np.random.default_rng(batch_seed)
    for frame in frames:
        for aoi in aois:
            shard = shard_dir / f"frame_{frame:04d}_aoi_{aoi}.npz"
            if shard.exists():
                with np.load(shard) as old:
                    count = len(old["x"])
                    pos = int(old["positive"].sum())
                samples += count
                positives += pos
                negatives += count - pos
                completed += 1
                continue
            images = data.centered_stack("train", frame, aoi, 5)
            truth = data.truth_xy("train", frame, aoi, moving_only=True)
            height, width = images[-1].shape
            values = np.stack(images).astype(np.float32) / 255.0
            with torch.inference_mode():
                scoremap = cluster(torch.from_numpy(values[None]).to(device))[0, 0].cpu().numpy()
            threshold = float(np.quantile(scoremap, cluster_quantile))
            from clusternet.postprocess import extract_roobis
            proposals = extract_roobis(scoremap, threshold)
            origin_scores: dict[tuple[int, int], float] = {}
            for proposal in proposals:
                for origin in origins_cover_bounds(proposal.input_bounds, width, height):
                    origin_scores[origin] = max(origin_scores.get(origin, 0.0), proposal.peak_score)
            positive_rows = [(origin, score) for origin, score in origin_scores.items() if patch_contains_truth(origin, truth)]
            negative_rows = [(origin, score) for origin, score in origin_scores.items() if not patch_contains_truth(origin, truth)]
            # Actual high-response false proposals are the most useful hard negatives.
            negative_rows.sort(key=lambda item: item[1], reverse=True)
            maximum_negatives = int(round(len(positive_rows) * negative_ratio))
            if not positive_rows:
                maximum_negatives = min(8, len(negative_rows))
            # Dense WAMI AOIs may not provide enough false ROOBIs. Fill the bank
            # with uniformly sampled background while retaining the real false
            # proposals as the hard half of the negative set.
            existing = {origin for origin, _ in origin_scores.items()}
            attempts = 0
            while len(negative_rows) < maximum_negatives and attempts < max(100, maximum_negatives * 40):
                attempts += 1
                origin = (
                    int(rng.integers(0, max(1, width - 127))),
                    int(rng.integers(0, max(1, height - 127))),
                )
                if origin in existing or patch_contains_truth(origin, truth):
                    continue
                existing.add(origin)
                sx = int(np.clip(round((origin[0] + 64 - 4.5) / 16), 0, scoremap.shape[1] - 1))
                sy = int(np.clip(round((origin[1] + 64 - 4.5) / 16), 0, scoremap.shape[0] - 1))
                negative_rows.append((origin, float(scoremap[sy, sx])))
            if len(negative_rows) > maximum_negatives:
                hard_count = maximum_negatives // 2
                hard = negative_rows[:hard_count]
                pool = negative_rows[hard_count:]
                random_count = maximum_negatives - len(hard)
                indexes = rng.choice(len(pool), size=random_count, replace=False) if random_count else []
                negative_rows = hard + [pool[int(index)] for index in indexes]
            rows = [(origin, score, True) for origin, score in positive_rows]
            rows += [(origin, score, False) for origin, score in negative_rows]
            rng.shuffle(rows)
            np.savez_compressed(
                shard,
                x=np.asarray([row[0][0] for row in rows], dtype=np.uint16),
                y=np.asarray([row[0][1] for row in rows], dtype=np.uint16),
                positive=np.asarray([row[2] for row in rows], dtype=np.bool_),
                score=np.asarray([row[1] for row in rows], dtype=np.float16),
            )
            count = len(rows)
            pos = len(positive_rows)
            samples += count
            positives += pos
            negatives += count - pos
            completed += 1
            print(json.dumps({"split": split_name, "frame": frame, "aoi": aoi, "samples": count, "positive": pos}), flush=True)
    return {
        "split": split_name,
        "frames": [frames.start, frames.stop - 1],
        "aois": aois,
        "shards": completed,
        "samples": samples,
        "positives": positives,
        "negatives": negatives,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache real ClusterNet ROOBI coordinates for FoveaNet-v2 training.")
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--cluster-checkpoint", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aois", nargs="+", default=list(AOIS))
    parser.add_argument("--train-frames", type=int, nargs=2, default=[102, 560])
    parser.add_argument("--val-frames", type=int, nargs=2, default=[561, 609])
    parser.add_argument("--cluster-quantile", type=float, default=0.90)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    add_external_repo(args.external_repo)
    from clusternet.model import ClusterNet
    from clusternet.real_data import CachedClusterNetData
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the ROOBI bank")
    checkpoint = torch.load(args.cluster_checkpoint, map_location="cpu", weights_only=False)
    cluster = ClusterNet(5).cuda().eval()
    cluster.load_state_dict(checkpoint["cluster"])
    args.output.mkdir(parents=True, exist_ok=True)
    with CachedClusterNetData(args.frame_cache, args.truth_cache) as data:
        reports = [
            build_split(
                split_name="train", frames=range(args.train_frames[0], args.train_frames[1] + 1),
                aois=args.aois, output=args.output, data=data, cluster=cluster,
                device=torch.device("cuda:0"), cluster_quantile=args.cluster_quantile,
                negative_ratio=args.negative_ratio, batch_seed=args.seed,
            ),
            build_split(
                split_name="val", frames=range(args.val_frames[0], args.val_frames[1] + 1),
                aois=args.aois, output=args.output, data=data, cluster=cluster,
                device=torch.device("cuda:0"), cluster_quantile=args.cluster_quantile,
                negative_ratio=args.negative_ratio, batch_seed=args.seed + 1,
            ),
        ]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "cluster_checkpoint": str(args.cluster_checkpoint.resolve()),
        "frame_cache": str(args.frame_cache.resolve()),
        "truth_cache": str(args.truth_cache.resolve()),
        "cluster_quantile": args.cluster_quantile,
        "negative_ratio": args.negative_ratio,
        "splits": reports,
    }
    (args.output / "bank.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
