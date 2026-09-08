import unittest

import numpy as np

from common.geometry.audit_truth_coordinates import _error_summary, _sample_frames


class TruthCoordinateAuditTests(unittest.TestCase):
    def test_sample_frames(self) -> None:
        self.assertEqual(_sample_frames(list(range(100, 612))), [100, 356, 611])

    def test_error_summary(self) -> None:
        summary = _error_summary(np.asarray([[0.0, 0.0], [0.6, 0.8]]))
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["max_norm"], 1.0)
        self.assertEqual(summary["fraction_within_1px"], 1.0)


if __name__ == "__main__":
    unittest.main()
