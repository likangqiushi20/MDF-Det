"""Detection and identity-aware metrics for saved HMRN track rows."""

from __future__ import annotations

from collections import Counter

from common.metrics.point_matching import match_points
from hmrn.targets import TrackedPoint
from hmrn.tracking import TrackPoint


def evaluate_frame(
    predictions: list[TrackPoint],
    truths: list[TrackedPoint],
    *,
    radius: float = 10,
) -> dict:
    matched = match_points(
        [(item.x, item.y) for item in predictions],
        [(item.x, item.y) for item in truths],
        radius,
    )
    return {
        "tp": matched.true_positives,
        "fp": matched.false_positives,
        "fn": matched.false_negatives,
        "matches": matched.matches,
        "distance_sum": sum(item[2] for item in matched.matches),
        "matched_id_pairs": [
            (predictions[prediction_index].track_id, truths[truth_index].track_id)
            for prediction_index, truth_index, _distance in matched.matches
        ],
    }


def summarize_sequence(frame_results: list[dict]) -> dict:
    tp = sum(item["tp"] for item in frame_results)
    fp = sum(item["fp"] for item in frame_results)
    fn = sum(item["fn"] for item in frame_results)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    distance_sum = sum(item.get("distance_sum", 0.0) for item in frame_results)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0,
        "mean_match_distance_px": distance_sum / tp if tp else None,
        "identity_switches": identity_switches(
            [item.get("matched_id_pairs", []) for item in frame_results]
        ),
    }


def identity_switches(
    matched_id_pairs_by_frame: list[list[tuple[int, int]]],
) -> int:
    """Count predicted-ID changes for each ground-truth ID across matched frames."""
    last_prediction: dict[int, int] = {}
    switches = 0
    for pairs in matched_id_pairs_by_frame:
        # Resolve accidental duplicate truth IDs deterministically.
        for prediction_id, truth_id in Counter(pairs):
            if truth_id in last_prediction and last_prediction[truth_id] != prediction_id:
                switches += 1
            last_prediction[truth_id] = prediction_id
    return switches
