"""Build a separate fixed-grid file for fair cross-paper comparison."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path


def build_fair_grids(
    fixed_grids_path: Path, protocol_path: Path, output_path: Path
) -> Path:
    fixed = json.loads(fixed_grids_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_name = protocol["source_grid_profile"]
    source = fixed["papers"][source_name]
    aois = protocol["aois"]
    common_grids = {aoi: copy.deepcopy(source["grids"][aoi]) for aoi in aois}

    papers = {}
    for paper in protocol["target_papers"]:
        papers[paper] = {
            "level": protocol["image_level"],
            "grid_status": "fair_comparison_common_grid",
            "center_basis": f"identical to {source_name} common comparison grid",
            "size_basis": f"identical to {source_name} common comparison grid",
            "grids": copy.deepcopy(common_grids),
        }

    output = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path.resolve()),
        "source_fixed_grids": str(fixed_grids_path.resolve()),
        "comparison_name": protocol["name"],
        "nominal_gsd_m_by_level": fixed["nominal_gsd_m_by_level"],
        "papers": papers,
        "warning": (
            "Use this file only for cross-paper comparison. Paper-native grid "
            "results must be generated with fixed_grid_profiles.json and reported separately."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-grids", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_fair_grids(args.fixed_grids, args.protocol, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
