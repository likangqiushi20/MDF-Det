"""Parse WPAFB truth CSV files and apply explicit, versioned policies."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterable, Iterator, Literal


TruthType = Literal["M", "I", "O"]
MotionRule = Literal["adjacent_any_motion", "legacy_static_pair_exclusion"]


@dataclass(frozen=True)
class TruthPoint:
    track_id: int
    latitude: float
    longitude: float
    x: float
    y: float
    timestamp: str
    frame_number: int
    truth_type: str


def haversine_m(first: TruthPoint, second: TruthPoint) -> float:
    radius_m = 6_371_008.8
    phi1 = math.radians(first.latitude)
    phi2 = math.radians(second.latitude)
    delta_phi = math.radians(second.latitude - first.latitude)
    delta_lambda = math.radians(second.longitude - first.longitude)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_m * math.asin(math.sqrt(value))


def iter_truth_csv(path: Path) -> Iterator[TruthPoint]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "id",
            "LATITUDE",
            "LONGITUDE",
            "X",
            "Y",
            "TIME",
            "FRAME_NUMBER",
            "TYPE",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Truth CSV is missing columns {sorted(missing)}: {path}")
        for row in reader:
            yield TruthPoint(
                track_id=int(row["id"]),
                latitude=float(row["LATITUDE"]),
                longitude=float(row["LONGITUDE"]),
                x=float(row["X"]),
                y=float(row["Y"]),
                timestamp=row["TIME"],
                frame_number=int(row["FRAME_NUMBER"]),
                truth_type=row["TYPE"].strip().upper(),
            )


def load_truth_csv(path: Path) -> list[TruthPoint]:
    return list(iter_truth_csv(path))


def iter_truth_tracks(path: Path) -> Iterator[list[TruthPoint]]:
    """Yield one track at a time without loading the full million-row CSV."""
    previous_track_id = -1
    for track_id, rows in groupby(iter_truth_csv(path), key=lambda point: point.track_id):
        if track_id < previous_track_id:
            raise ValueError(
                f"Truth CSV is not grouped by non-decreasing track ID: {path}"
            )
        previous_track_id = track_id
        yield list(rows)


@dataclass(frozen=True)
class TruthPolicy:
    """A reproducible truth selection policy.

    ``adjacent_any_motion`` marks a point as moving when either adjacent
    consecutive-frame interval moves at least ``minimum_motion_m``. This
    matches ``compute_metrics.py`` in the current project while adding an
    explicit consecutive-frame check.

    ``legacy_static_pair_exclusion`` mirrors ``WAMI_detector_multi_AOI.py``:
    every pair below the threshold invalidates both endpoints. It is retained
    only to reproduce legacy results because it can remove a moving point next
    to a temporary stop.
    """

    name: str
    allowed_types: frozenset[str] = frozenset({"M", "I", "O"})
    moving_only: bool = True
    minimum_motion_m: float = 0.8
    motion_rule: MotionRule = "adjacent_any_motion"
    require_consecutive_frames: bool = True

    def apply(self, points: Iterable[TruthPoint]) -> list[TruthPoint]:
        selected = [point for point in points if point.truth_type in self.allowed_types]
        if not self.moving_only:
            return selected
        if self.minimum_motion_m < 0:
            raise ValueError("minimum_motion_m must be non-negative")

        tracks: dict[int, list[TruthPoint]] = defaultdict(list)
        for point in selected:
            tracks[point.track_id].append(point)

        keep: set[tuple[int, int]] = set()
        if self.motion_rule == "adjacent_any_motion":
            for track_id, track in tracks.items():
                ordered = sorted(track, key=lambda point: point.frame_number)
                for first, second in zip(ordered, ordered[1:]):
                    if self.require_consecutive_frames and second.frame_number != first.frame_number + 1:
                        continue
                    if haversine_m(first, second) >= self.minimum_motion_m:
                        keep.add((track_id, first.frame_number))
                        keep.add((track_id, second.frame_number))
        elif self.motion_rule == "legacy_static_pair_exclusion":
            keep = {(point.track_id, point.frame_number) for point in selected}
            for track_id, track in tracks.items():
                ordered = sorted(track, key=lambda point: point.frame_number)
                for first, second in zip(ordered, ordered[1:]):
                    if self.require_consecutive_frames and second.frame_number != first.frame_number + 1:
                        continue
                    if haversine_m(first, second) < self.minimum_motion_m:
                        keep.discard((track_id, first.frame_number))
                        keep.discard((track_id, second.frame_number))
        else:
            raise ValueError(f"Unsupported motion rule: {self.motion_rule}")

        return [
            point
            for point in selected
            if (point.track_id, point.frame_number) in keep
        ]
