from pathlib import Path
import tempfile
import unittest

from common.labels.truth import TruthPoint, TruthPolicy, iter_truth_tracks, load_truth_csv


def point(frame: int, longitude: float, truth_type: str = "M") -> TruthPoint:
    return TruthPoint(
        track_id=1,
        latitude=0.0,
        longitude=longitude,
        x=0.0,
        y=0.0,
        timestamp="",
        frame_number=frame,
        truth_type=truth_type,
    )


class TruthTests(unittest.TestCase):
    def test_csv_parser(self) -> None:
        content = (
            "id,LATITUDE,LONGITUDE,X,Y,TIME,FRAME_NUMBER,TYPE\n"
            "7,39.0,-84.0,10,20,t,100,I\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.csv"
            path.write_text(content, encoding="utf-8")
            parsed = load_truth_csv(path)
        self.assertEqual(parsed[0].track_id, 7)
        self.assertEqual(parsed[0].frame_number, 100)
        self.assertEqual(parsed[0].truth_type, "I")

    def test_track_streaming(self) -> None:
        content = (
            "id,LATITUDE,LONGITUDE,X,Y,TIME,FRAME_NUMBER,TYPE\n"
            "1,0,0,0,0,t,1,M\n"
            "1,0,0,0,0,t,2,M\n"
            "2,0,0,0,0,t,1,I\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.csv"
            path.write_text(content, encoding="utf-8")
            tracks = list(iter_truth_tracks(path))
        self.assertEqual([len(track) for track in tracks], [2, 1])
        self.assertEqual([track[0].track_id for track in tracks], [1, 2])

    def test_persistent_policy_keeps_allowed_types(self) -> None:
        policy = TruthPolicy(
            name="persistent_manual",
            allowed_types=frozenset({"M"}),
            moving_only=False,
        )
        self.assertEqual(policy.apply([point(1, 0, "M"), point(2, 0, "I")]), [point(1, 0, "M")])

    def test_adjacent_any_motion_keeps_endpoints_of_moving_interval(self) -> None:
        stationary = point(1, 0.0)
        transition = point(2, 0.0)
        moving = point(3, 0.00002)
        policy = TruthPolicy(name="moving")
        self.assertEqual(policy.apply([stationary, transition, moving]), [transition, moving])

    def test_legacy_rule_removes_point_next_to_stop(self) -> None:
        points = [point(1, 0.0), point(2, 0.0), point(3, 0.00002)]
        policy = TruthPolicy(
            name="legacy",
            motion_rule="legacy_static_pair_exclusion",
        )
        self.assertEqual(policy.apply(points), [points[2]])

    def test_frame_gap_is_not_treated_as_adjacent_motion(self) -> None:
        policy = TruthPolicy(name="moving")
        self.assertEqual(policy.apply([point(1, 0.0), point(3, 0.0001)]), [])


if __name__ == "__main__":
    unittest.main()
