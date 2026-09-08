"""Exact output/input geometry for the reconstructed two-stage networks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReceptiveGeometry:
    jump: float
    receptive_field: float
    first_center: float

    def input_to_output(self, coordinate: float) -> float:
        return (coordinate - self.first_center) / self.jump

    def output_to_input(self, coordinate: float) -> float:
        return self.first_center + self.jump * coordinate


CLUSTERNET_GEOMETRY = ReceptiveGeometry(
    jump=16.0,
    receptive_field=37.0,
    first_center=4.5,
)
FOVEANET_GEOMETRY = ReceptiveGeometry(
    jump=2.0,
    receptive_field=100.0,
    first_center=49.5,
)


def roobi_input_bounds(
    output_left: int,
    output_top: int,
    output_right: int,
    output_bottom: int,
) -> tuple[int, int, int, int]:
    """Return inclusive-exclusive input bounds covering output receptive fields."""
    half = CLUSTERNET_GEOMETRY.receptive_field / 2
    left = int(CLUSTERNET_GEOMETRY.output_to_input(output_left) - half)
    top = int(CLUSTERNET_GEOMETRY.output_to_input(output_top) - half)
    right = int(
        CLUSTERNET_GEOMETRY.output_to_input(output_right) + half + 1
    )
    bottom = int(
        CLUSTERNET_GEOMETRY.output_to_input(output_bottom) + half + 1
    )
    return left, top, right, bottom
