import unittest

from common.data.fixed_grid_truth import project_truth_points
from common.geometry.fixed_grids import _grid_from_center
from common.labels.truth import TruthPoint


class FixedGridTruthTests(unittest.TestCase):
    def test_center_point_projects_inside(self) -> None:
        grid = _grid_from_center(-84.0, 39.0, 101, 101, 0.5)
        point = TruthPoint(1, 39.0, -84.0, 0, 0, "", 100, "M")
        projected = project_truth_points([point], grid)
        self.assertEqual(len(projected), 1)
        self.assertAlmostEqual(projected[0].x, 50.0, places=5)
        self.assertAlmostEqual(projected[0].y, 50.0, places=5)

    def test_far_point_is_excluded(self) -> None:
        grid = _grid_from_center(-84.0, 39.0, 101, 101, 0.5)
        point = TruthPoint(1, 40.0, -84.0, 0, 0, "", 100, "M")
        self.assertEqual(project_truth_points([point], grid), [])


if __name__ == "__main__":
    unittest.main()
