"""Compare canonical and legacy moving-truth policies by AOI."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from common.labels.truth import TruthPoint, TruthPolicy, iter_truth_tracks


def _inside(point: TruthPoint, bounds: dict[str, float]) -> bool:
    return (
        bounds["left"] <= point.longitude <= bounds["right"]
        and bounds["bottom"] <= point.latitude <= bounds["top"]
    )


def _increment(
    counter: Counter[str],
    points: list[TruthPoint],
    geographic: dict[str, dict[str, Any]],
) -> None:
    counter["all"] += len(points)
    for point in points:
        for aoi_name, bounds in geographic.items():
            if "left" in bounds and _inside(point, bounds):
                counter[aoi_name] += 1


def build_truth_policy_audit(
    manifest_path: Path,
    profiles_path: Path,
    output_path: Path,
    level: str = "r1",
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    dataset_root = Path(manifest["dataset_root"])
    geographic = profiles["aoi_profiles"]["geographic_bounds_wgs84"]
    policies = {
        "persistent_all_types": TruthPolicy(
            name="persistent_all_types",
            moving_only=False,
        ),
        "canonical_adjacent_any_motion": TruthPolicy(
            name="canonical_adjacent_any_motion",
            moving_only=True,
            minimum_motion_m=0.8,
            motion_rule="adjacent_any_motion",
            require_consecutive_frames=True,
        ),
        "legacy_static_pair_exclusion": TruthPolicy(
            name="legacy_static_pair_exclusion",
            moving_only=True,
            minimum_motion_m=0.8,
            motion_rule="legacy_static_pair_exclusion",
            require_consecutive_frames=False,
        ),
    }

    results = {}
    for frame_set in ("train", "self_test"):
        truth_relative = manifest["frame_sets"][frame_set]["truth_files"][level]
        counters = {name: Counter() for name in policies}
        tracks = 0
        for track in iter_truth_tracks(dataset_root / truth_relative):
            tracks += 1
            for name, policy in policies.items():
                _increment(counters[name], policy.apply(track), geographic)
        results[frame_set] = {
            "truth_path": truth_relative,
            "track_count": tracks,
            "counts": {name: dict(counter) for name, counter in counters.items()},
        }

    output = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "profiles": str(profiles_path.resolve()),
        "level": level,
        "policies": {
            "canonical_adjacent_any_motion": (
                "Keep a point if either consecutive neighboring interval moves "
                "at least 0.8 m."
            ),
            "legacy_static_pair_exclusion": (
                "Mirror WAMI_detector_multi_AOI.py: any below-threshold next-row "
                "pair removes both endpoints, without a frame-gap check."
            ),
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", default="r1")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = build_truth_policy_audit(
        args.manifest,
        args.profiles,
        args.output,
        level=args.level,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
