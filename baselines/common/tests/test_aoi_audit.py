import unittest

from rasterio.windows import Window

from common.geometry.audit_aois import _round_window


class AoiAuditTests(unittest.TestCase):
    def test_round_window_contains_fractional_window(self) -> None:
        rounded = _round_window(Window(10.2, 20.8, 30.1, 40.1))
        self.assertEqual(rounded, Window(10, 20, 31, 41))


if __name__ == "__main__":
    unittest.main()
