"""Run matched hard-negative and dilation ablations for CATLoss."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from catloss.train import train


VARIANTS = {
    "no_hard_negatives": {"hard_negative_fraction": 0.0},
    "full_dilated": {"dilation_mode": "full"},
    "augmentation": {"augmentation": True},
    "augmentation_reflect": {"augmentation": True},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    config_dir = CODE_ROOT / "generated" / "ablation_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.variants:
        config = dict(base)
        config.update(VARIANTS[name])
        config.update(
            name=f"catloss_architecture_ablation_{name}",
            epochs=args.epochs,
            checkpoint=f"runs/catloss/ablation_{name}.pt",
            report=f"generated/catloss_ablation_{name}_train.json",
        )
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        report = train(path, args.manifest, args.fixed_grids, args.truth_cache)
        reports.append(
            {
                "variant": name,
                "best_epoch": report["best_epoch"],
                "elapsed_seconds": report["elapsed_seconds"],
                "checkpoint": report["checkpoint"],
            }
        )
    summary = {
        "schema_version": 1,
        "status": "ok",
        "self_test_used": False,
        "runs": reports,
    }
    output = CODE_ROOT / "generated" / "catloss_architecture_ablation_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
