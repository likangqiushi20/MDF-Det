import unittest

from common.geometry.fixed_grids import _grid_from_center, _half_size


class FixedGridTests(unittest.TestCase):
    def test_half_size_rounds_up(self) -> None:
        self.assertEqual(_half_size([3265, 2542]), [1633, 1271])

    def test_grid_is_centered_and_has_requested_size(self) -> None:
        grid = _grid_from_center(-84.0, 39.0, 100, 200, 0.5)
        self.assertEqual((grid["width"], grid["height"]), (100, 200))
        self.assertAlmostEqual(
            (grid["bounds"]["left"] + grid["bounds"]["right"]) / 2.0,
            -84.0,
        )
        self.assertAlmostEqual(
            (grid["bounds"]["bottom"] + grid["bounds"]["top"]) / 2.0,
            39.0,
        )


if __name__ == "__main__":
    unittest.main()
