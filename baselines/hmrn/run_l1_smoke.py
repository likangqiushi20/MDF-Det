"""Run a short real-data CUDA overfit smoke test for HMRN."""

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

from hmrn.data import build_training_sample
from hmrn.losses import CenterFocalLoss, masked_displacement_l1
from hmrn.model import HMRN
from hmrn.real_data import HMRNData


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Optional smoke-only checkpoint for exercising inference/evaluation.",
    )
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(7)
    np.random.seed(7)
    device = torch.device("cuda:0")
    with HMRNData(
        args.manifest, args.fixed_grids, args.truth_cache
    ) as data:
        pair = data.read_pair("train", 101, "01")
    sample = build_training_sample(*pair, np.random.default_rng(7))
    inputs = sample.inputs[None].to(device)
    center = sample.center[None].to(device)
    motion = sample.displacement[None].to(device)
    mask = sample.displacement_mask[None].to(device)
    model = HMRN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000125)
    focal = CenterFocalLoss()
    history = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        center_loss = focal(output["center"], center)
        motion_loss = masked_displacement_l1(output["motion"], motion, mask)
        loss = center_loss + motion_loss
        loss.backward()
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "total": float(loss.detach()),
                "center": float(center_loss.detach()),
                "motion": float(motion_loss.detach()),
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
        "input_shape": list(inputs.shape),
        "center_shape": list(center.shape),
        "objects_in_crop": int(center.eq(1).sum()),
        "tracked_objects_in_crop": int(mask.sum()),
        "steps": args.steps,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "test_set_used": False,
    }
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "epoch": 0,
                "smoke_only": True,
            },
            args.checkpoint,
        )
        result["checkpoint"] = str(args.checkpoint.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
