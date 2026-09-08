"""Run matched 1/2/3-channel temporal CATLoss training ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from catloss.train import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--channels", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    runs = []
    config_dir = CODE_ROOT / "generated" / "ablation_configs"
    for channels in args.channels:
        config = dict(base)
        config.update(
            name=f"catloss_temporal_{channels}",
            temporal_channels=channels,
            bgs_temporal_channels=4,
            epochs=args.epochs,
            checkpoint=f"runs/catloss/ablation_temporal_{channels}.pt",
            report=f"generated/catloss_ablation_temporal_{channels}_train.json",
        )
        path = config_dir / f"temporal_{channels}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        report = train(path, args.manifest, args.fixed_grids, args.truth_cache)
        runs.append({"channels": channels, "best_epoch": report["best_epoch"], "elapsed_seconds": report["elapsed_seconds"], "checkpoint": report["checkpoint"]})
    summary = {"schema_version": 1, "status": "ok", "self_test_used": False, "runs": runs}
    output = CODE_ROOT / "generated" / "catloss_temporal_ablation_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
