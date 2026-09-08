"""ClusterNet-specific five-frame moving-object truth policy."""

from __future__ import annotations

from typing import Iterable

from common.labels.truth import TruthPoint, haversine_m


def five_frame_motion_keys(
    points: Iterable[TruthPoint],
    *,
    minimum_displacement_m: float = 3.75,
) -> set[tuple[int, int]]:
    """Keep points participating in a consecutive five-frame moving window.

    The paper removes vehicles moving fewer than 15 r0 pixels over five
    frames. At the audited ~0.25 m/px resolution this is 3.75 m. The paper
    does not define endpoint handling; this reproduction marks all five
    samples in any qualifying consecutive window.
    """
    ordered = sorted(points, key=lambda point: point.frame_number)
    keep: set[tuple[int, int]] = set()
    for start in range(max(0, len(ordered) - 4)):
        window = ordered[start : start + 5]
        if len(window) != 5:
            continue
        if any(
            second.frame_number != first.frame_number + 1
            for first, second in zip(window, window[1:])
        ):
            continue
        if haversine_m(window[0], window[-1]) >= minimum_displacement_m:
            keep.update((point.track_id, point.frame_number) for point in window)
    return keep
