from hmrn.evaluate import evaluate_frame, identity_switches, summarize_sequence
from hmrn.targets import TrackedPoint
from hmrn.tracking import TrackPoint


def test_sequence_summary_and_identity_switch_count():
    summary = summarize_sequence(
        [
            {"tp": 2, "fp": 1, "fn": 0},
            {"tp": 1, "fp": 0, "fn": 1},
        ]
    )
    assert summary["precision"] == 0.75
    assert summary["recall"] == 0.75
    assert identity_switches([[(4, 9)], [(5, 9)]]) == 1


def test_frame_evaluation_carries_distances_and_identity_pairs():
    frame = evaluate_frame(
        [TrackPoint(4, 10, 10, 0.9), TrackPoint(5, 100, 100, 0.8)],
        [TrackedPoint(9, 13, 14)],
        radius=10,
    )
    assert (frame["tp"], frame["fp"], frame["fn"]) == (1, 1, 0)
    assert frame["distance_sum"] == 5
    assert frame["matched_id_pairs"] == [(4, 9)]
    summary = summarize_sequence([frame])
    assert summary["mean_match_distance_px"] == 5
    assert summary["identity_switches"] == 0
