"""Exercise real CATLoss truth caching, hard-negative mining, and GPU batches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from catloss.data import (
    build_localization_batch,
    build_patch_pools,
    sample_objectness_epoch,
)
from catloss.losses import BinaryFocalLoss, CrowdAwareThresholdedLoss
from catloss.model import LocalizationNetwork, ObjectnessNetwork
from catloss.real_data import CATLossData


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.perf_counter()
    rng = np.random.default_rng(17)
    with CATLossData(args.manifest, args.fixed_grids, args.truth_cache) as data:
        images = data.read_causal_stack("train", args.frame, args.aoi, channels=4)
        truth_xy = data.truth_xy("train", args.frame, args.aoi)
    pools = build_patch_pools(images, truth_xy)
    object_x, object_y = sample_objectness_epoch(
        images,
        pools,
        rng,
        negatives_per_positive=1,
        hard_fraction=0.5,
    )
    localization_centers = [
        center
        for center in pools.positive
        if 22 <= center[0] < images[-1].shape[1] - 22
        and 22 <= center[1] < images[-1].shape[0] - 22
    ]
    local_x, local_y = build_localization_batch(
        images, truth_xy, localization_centers
    )

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    object_model = ObjectnessNetwork().to(device).train()
    local_model = LocalizationNetwork(dilation_mode="half").to(device).train()
    object_loss = BinaryFocalLoss()(
        object_model(torch.from_numpy(object_x).to(device)),
        torch.from_numpy(object_y).to(device),
    )
    local_loss = CrowdAwareThresholdedLoss(tau=0.2, q=0.5)(
        local_model(torch.from_numpy(local_x).to(device)),
        torch.from_numpy(local_y).to(device),
    )
    (object_loss + local_loss).backward()

    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "device": torch.cuda.get_device_name(0),
        "split": "train",
        "frame": args.frame,
        "aoi": args.aoi,
        "truth_count": len(truth_xy),
        "pools": {
            "positive": len(pools.positive),
            "hard_negative": len(pools.hard_negative),
            "ordinary_negative": len(pools.ordinary_negative),
        },
        "batches": {
            "objectness": list(object_x.shape),
            "localization": list(local_x.shape),
        },
        "losses": {
            "objectness_focal": float(object_loss.detach()),
            "localization_catloss": float(local_loss.detach()),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=103)
    parser.add_argument("--aoi", default="01")
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
