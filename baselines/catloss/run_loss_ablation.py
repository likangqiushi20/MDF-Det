"""Run matched MSE, TLoss, and CATLoss training jobs from one base config."""

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
    parser.add_argument("--losses", nargs="+", default=["mse", "tloss", "catloss"])
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    reports = []
    generated_configs = CODE_ROOT / "generated" / "ablation_configs"
    generated_configs.mkdir(parents=True, exist_ok=True)
    for loss_name in args.losses:
        config = dict(base)
        config["name"] = f"catloss_loss_ablation_{loss_name}"
        config["epochs"] = args.epochs
        config["localization_loss"] = loss_name
        config["checkpoint"] = f"runs/catloss/ablation_{loss_name}.pt"
        config["report"] = f"generated/catloss_ablation_{loss_name}_train.json"
        config_path = generated_configs / f"{loss_name}.json"
        config_path.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(
            train(
                config_path,
                args.manifest,
                args.fixed_grids,
                args.truth_cache,
            )
        )
    summary = {
        "schema_version": 1,
        "status": "ok",
        "self_test_used": False,
        "runs": [
            {
                "loss": report["localization_loss"],
                "best_epoch": report["best_epoch"],
                "elapsed_seconds": report["elapsed_seconds"],
                "checkpoint": report["checkpoint"],
            }
            for report in reports
        ],
    }
    output = CODE_ROOT / "generated" / "catloss_loss_ablation_train_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
