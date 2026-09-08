"""Run a short real-data CUDA HM-Net overfit smoke."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from hm_net.data import HMNetData, build_training_sample
from hm_net.losses import hmnet_loss
from hm_net.model import HMNet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(11)
    device = torch.device("cuda:0")
    with HMNetData(
        args.manifest, args.fixed_grids, args.truth_cache
    ) as data:
        pair = data.read_pair("train", 101, "01")
    sample = build_training_sample(*pair, np.random.default_rng(11))
    tensors = {
        name: getattr(sample, name)[None].to(device)
        for name in (
            "current",
            "previous",
            "feedback",
            "center",
            "motion",
            "precision",
            "motion_mask",
            "precision_mask",
        )
    }
    model = HMNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0013)
    history = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(
            tensors["current"], tensors["previous"], tensors["feedback"]
        )
        loss, parts = hmnet_loss(
            predictions,
            tensors["center"],
            tensors["motion"],
            tensors["precision"],
            tensors["motion_mask"],
            tensors["precision_mask"],
        )
        loss.backward()
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "total": float(loss.detach()),
                **{name: float(value.detach()) for name, value in parts.items()},
            }
        )
    torch.cuda.synchronize()
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "device": torch.cuda.get_device_name(0),
        "split": "train",
        "frame_pair": [100, 101],
        "aoi": "01",
        "input_shapes": {
            name: list(tensors[name].shape)
            for name in ("current", "previous", "feedback")
        },
        "moving_centers": int(tensors["center"][:, 0].eq(1).sum()),
        "stationary_centers": int(tensors["center"][:, 1].eq(1).sum()),
        "steps": args.steps,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "test_set_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
