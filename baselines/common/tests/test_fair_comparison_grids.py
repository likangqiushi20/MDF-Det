import json
from pathlib import Path

from common.geometry.build_fair_comparison_grids import build_fair_grids


def test_all_methods_receive_identical_grids(tmp_path: Path):
    fixed = {
        "nominal_gsd_m_by_level": {"r1": 0.5},
        "papers": {
            "hmrn": {
                "level": "r1",
                "grids": {
                    "01": {"width": 10, "height": 8, "transform": [1, 0, 0, 0, -1, 0]}
                },
            }
        },
    }
    protocol = {
        "name": "fair",
        "source_grid_profile": "hmrn",
        "target_papers": ["a", "b"],
        "image_level": "r1",
        "aois": ["01"],
    }
    fixed_path = tmp_path / "fixed.json"
    protocol_path = tmp_path / "protocol.json"
    output_path = tmp_path / "fair.json"
    fixed_path.write_text(json.dumps(fixed))
    protocol_path.write_text(json.dumps(protocol))

    build_fair_grids(fixed_path, protocol_path, output_path)
    result = json.loads(output_path.read_text())

    assert result["papers"]["a"]["level"] == "r1"
    assert result["papers"]["a"]["grids"] == result["papers"]["b"]["grids"]
    assert result["papers"]["a"]["grids"] is not result["papers"]["b"]["grids"]
