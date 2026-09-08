import unittest

from common.metrics.point_matching import match_points


class PointMatchingTests(unittest.TestCase):
    def test_empty_inputs(self) -> None:
        result = match_points([], [], radius=10)
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (0, 0, 0))

    def test_one_truth_cannot_match_twice(self) -> None:
        result = match_points([(0, 0), (1, 0)], [(0, 0)], radius=10)
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (1, 1, 0))
        self.assertEqual(result.matches[0][:2], (0, 0))

    def test_radius_is_inclusive(self) -> None:
        result = match_points([(3, 4)], [(0, 0)], radius=5)
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.matches[0][2], 5.0)

    def test_assignment_maximizes_valid_matches(self) -> None:
        result = match_points([(0, 0), (2, 0)], [(1, 0), (-1, 0)], radius=1.1)
        self.assertEqual(result.true_positives, 2)
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.false_negatives, 0)

    def test_outside_radius_is_unmatched(self) -> None:
        result = match_points([(0, 0)], [(10, 0)], radius=5)
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
