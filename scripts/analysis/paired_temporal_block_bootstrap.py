"""Paired temporally blocked bootstrap for MDF-Det versus one baseline.

Each input directory must contain ``aoi01.json``, ..., ``aoi41.json``.  Every
JSON file must have a ``per_frame`` list whose rows contain frame/tp/fp/fn.
The same temporal blocks are sampled for both methods and all AOIs.  Counts
are accumulated before F1 is recomputed; block-level F1 values are never
averaged directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


AOIS = ("01", "02", "03", "34", "40", "41")
COUNT_KEYS = ("tp", "fp", "fn")


def f1_from_counts(counts: np.ndarray) -> np.ndarray:
    """Return F1 along the final count axis [..., (tp, fp, fn)]."""
    tp, fp, fn = (counts[..., index].astype(np.float64) for index in range(3))
    denominator = 2.0 * tp + fp + fn
    return np.divide(2.0 * tp, denominator, out=np.zeros_like(denominator),
                     where=denominator > 0)


def load_method(folder: Path, first: int, last: int) -> np.ndarray:
    """Load [time, AOI, (tp, fp, fn)] counts."""
    frames = list(range(first, last + 1))
    output = np.zeros((len(frames), len(AOIS), 3), dtype=np.int64)
    for aoi_index, aoi in enumerate(AOIS):
        path = folder / f"aoi{aoi}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = {int(row["frame"]): row for row in payload.get("per_frame", [])}
        missing = [frame for frame in frames if frame not in rows]
        if missing:
            raise ValueError(f"{path} lacks {len(missing)} frames; first={missing[:5]}")
        for time_index, frame in enumerate(frames):
            output[time_index, aoi_index] = [int(rows[frame][key]) for key in COUNT_KEYS]
    return output


def metrics(counts: np.ndarray) -> dict:
    per_aoi_counts = counts.sum(axis=0)
    per_aoi_f1 = f1_from_counts(per_aoi_counts)
    pooled = f1_from_counts(per_aoi_counts.sum(axis=0))
    return {
        "macro_aoi_f1": float(per_aoi_f1.mean()),
        "pooled_f1": float(pooled),
        "per_aoi_f1": {aoi: float(value) for aoi, value in zip(AOIS, per_aoi_f1)},
        "tp": int(per_aoi_counts[:, 0].sum()),
        "fp": int(per_aoi_counts[:, 1].sum()),
        "fn": int(per_aoi_counts[:, 2].sum()),
    }


def complete_blocks(frame_count: int, block_size: int) -> list[np.ndarray]:
    block_count = frame_count // block_size
    return [np.arange(index * block_size, (index + 1) * block_size)
            for index in range(block_count)]


def bootstrap(mdf: np.ndarray, baseline: np.ndarray, block_size: int,
              resamples: int, seed: int) -> dict:
    blocks = complete_blocks(len(mdf), block_size)
    if len(blocks) < 2:
        raise ValueError(f"Block size {block_size} produces fewer than two blocks")
    used_count = len(blocks) * block_size
    # First sum frames within each block, retaining the AOI and count axes.
    mdf_blocks = np.stack([mdf[indexes].sum(axis=0) for indexes in blocks])
    baseline_blocks = np.stack([baseline[indexes].sum(axis=0) for indexes in blocks])
    rng = np.random.default_rng(seed + block_size)
    selected = rng.integers(0, len(blocks), size=(resamples, len(blocks)))

    macro_delta = np.empty(resamples, np.float64)
    pooled_delta = np.empty(resamples, np.float64)
    for start in range(0, resamples, 500):
        ids = selected[start:start + 500]
        mdf_counts = mdf_blocks[ids].sum(axis=1)          # [R, AOI, count]
        base_counts = baseline_blocks[ids].sum(axis=1)
        macro_delta[start:start + len(ids)] = (
            f1_from_counts(mdf_counts).mean(axis=1)
            - f1_from_counts(base_counts).mean(axis=1)
        )
        pooled_delta[start:start + len(ids)] = (
            f1_from_counts(mdf_counts.sum(axis=1))
            - f1_from_counts(base_counts.sum(axis=1))
        )

    observed_mdf = metrics(mdf[:used_count])
    observed_baseline = metrics(baseline[:used_count])
    return {
        "block_size": block_size,
        "complete_blocks": len(blocks),
        "used_frames": used_count,
        "dropped_tail_frames": len(mdf) - used_count,
        "observed_macro_aoi_f1_difference": (
            observed_mdf["macro_aoi_f1"] - observed_baseline["macro_aoi_f1"]),
        "macro_aoi_f1_difference_bootstrap_mean": float(macro_delta.mean()),
        "macro_aoi_f1_difference_95_ci": np.percentile(
            macro_delta, [2.5, 97.5]).tolist(),
        "observed_pooled_f1_difference": (
            observed_mdf["pooled_f1"] - observed_baseline["pooled_f1"]),
        "pooled_f1_difference_bootstrap_mean": float(pooled_delta.mean()),
        "pooled_f1_difference_95_ci": np.percentile(
            pooled_delta, [2.5, 97.5]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mdf-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-frame", type=int, default=612)
    parser.add_argument("--last-frame", type=int, default=1124)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=(10, 20, 30))
    parser.add_argument("--primary-block", type=int, default=20)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.primary_block not in args.block_sizes:
        parser.error("--primary-block must be included in --block-sizes")
    if args.last_frame < args.first_frame:
        parser.error("--last-frame must be >= --first-frame")

    mdf = load_method(args.mdf_dir, args.first_frame, args.last_frame)
    baseline = load_method(args.baseline_dir, args.first_frame, args.last_frame)
    # Both methods must be matched against exactly the same truth instances.
    mdf_truth = mdf[..., 0] + mdf[..., 2]
    baseline_truth = baseline[..., 0] + baseline[..., 2]
    if not np.array_equal(mdf_truth, baseline_truth):
        where = np.argwhere(mdf_truth != baseline_truth)[0]
        raise ValueError(
            "Methods do not share identical per-frame truth counts at "
            f"frame={args.first_frame + int(where[0])}, AOI={AOIS[int(where[1])]}"
        )

    full_mdf, full_baseline = metrics(mdf), metrics(baseline)
    analyses = [bootstrap(mdf, baseline, size, args.resamples, args.seed)
                for size in args.block_sizes]
    report = {
        "design": "paired non-overlapping temporal-block bootstrap",
        "comparison": "MDF-Det minus strongest baseline",
        "frames": [args.first_frame, args.last_frame],
        "frame_count": len(mdf),
        "aois": list(AOIS),
        "resamples": args.resamples,
        "seed": args.seed,
        "primary_block_size": args.primary_block,
        "primary_estimand": "difference in the mean of six AOI-specific F1 scores",
        "full_sequence": {
            "mdf_det": full_mdf,
            "baseline": full_baseline,
            "macro_aoi_f1_difference": full_mdf["macro_aoi_f1"] - full_baseline["macro_aoi_f1"],
            "pooled_f1_difference": full_mdf["pooled_f1"] - full_baseline["pooled_f1"],
        },
        "block_sensitivity": analyses,
        "notes": [
            "Identical block indices are used for both methods and all six AOIs.",
            "TP/FP/FN are accumulated before F1 is recomputed.",
            "Only complete non-overlapping blocks enter each confidence interval.",
            "The main table's macro-AOI F1 difference is the primary estimand; pooled F1 is secondary.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "paired_temporal_block_bootstrap.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "block_size_sensitivity.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        fields = list(analyses[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(analyses)
    primary = next(item for item in analyses if item["block_size"] == args.primary_block)
    full = report["full_sequence"]
    markdown = [
        "# Paired temporal-block bootstrap results",
        "",
        f"- Frames: {args.first_frame}--{args.last_frame} ({len(mdf)} frames per AOI)",
        f"- AOIs: {', '.join(AOIS)}",
        f"- Resamples: {args.resamples:,}",
        f"- Primary temporal block: {args.primary_block} frames",
        "- Difference: MDF-Det minus HM-Net",
        "",
        "## Full-sequence point estimates",
        "",
        "| Method | Macro-AOI F1 | Pooled F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|",
        (f"| MDF-Det | {full['mdf_det']['macro_aoi_f1']:.6f} | "
         f"{full['mdf_det']['pooled_f1']:.6f} | {full['mdf_det']['tp']} | "
         f"{full['mdf_det']['fp']} | {full['mdf_det']['fn']} |"),
        (f"| HM-Net | {full['baseline']['macro_aoi_f1']:.6f} | "
         f"{full['baseline']['pooled_f1']:.6f} | {full['baseline']['tp']} | "
         f"{full['baseline']['fp']} | {full['baseline']['fn']} |"),
        "",
        (f"Full-sequence macro-AOI F1 difference: "
         f"**{full['macro_aoi_f1_difference']:+.6f}**."),
        "",
        "## Block-size sensitivity",
        "",
        "| Block (frames) | Complete blocks | Used frames | Dropped tail | Macro-F1 difference | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analyses:
        low, high = row["macro_aoi_f1_difference_95_ci"]
        markdown.append(
            f"| {row['block_size']} | {row['complete_blocks']} | {row['used_frames']} | "
            f"{row['dropped_tail_frames']} | "
            f"{row['observed_macro_aoi_f1_difference']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    markdown.extend([
        "",
        "The primary analysis is the row with a 20-frame block. Counts are "
        "aggregated before F1 is recomputed, and identical temporal blocks are "
        "used for both methods and all AOIs.",
        "",
    ])
    (args.output / "PAIRED_BOOTSTRAP_RESULTS.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps({"full_sequence": report["full_sequence"],
                      "primary_analysis": primary}, indent=2))


if __name__ == "__main__":
    main()
