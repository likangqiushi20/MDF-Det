import unittest

import numpy as np
from affine import Affine

from common.geometry.coordinates import (
    geo_to_fractional_pixel_center,
    geo_to_pixel_index,
    pixel_index_to_geo,
)
from common.geometry.registration import transform_points_source_to_destination


class CoordinateTests(unittest.TestCase):
    def test_pixel_center_round_trip(self) -> None:
        transform = Affine(2.0, 0.0, 10.0, 0.0, -2.0, 20.0)
        longitude, latitude = pixel_index_to_geo(transform, 3, 4)
        self.assertEqual((longitude, latitude), (17.0, 11.0))
        self.assertEqual(geo_to_pixel_index(transform, longitude, latitude), (3, 4))
        x, y = geo_to_fractional_pixel_center(transform, longitude, latitude)
        self.assertAlmostEqual(x, 3.0)
        self.assertAlmostEqual(y, 4.0)

    def test_source_to_destination_homography(self) -> None:
        homography = np.asarray(
            [[1.0, 0.0, 5.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]
        )
        transformed = transform_points_source_to_destination([(1.0, 3.0)], homography)
        np.testing.assert_allclose(transformed, [[6.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
