import numpy as np

from hmrn.data import build_training_sample, select_crop_origin
from hmrn.infer import make_input
from hmrn.targets import TrackedPoint
from hmrn.tracking import TrackPoint


def test_crop_origin_is_clamped_and_contains_edge_anchor():
    rng = np.random.default_rng(2)
    assert select_crop_origin(
        (600, 1000), [TrackedPoint(1, 990, 590)], rng
    ) == (40, 56)


def test_training_sample_has_paper_shapes_and_previous_heatmap():
    previous = np.zeros((600, 1000), dtype=np.uint8)
    current = np.zeros_like(previous)
    current[300, 500] = 255
    points_previous = [TrackedPoint(8, 496, 304)]
    points_current = [TrackedPoint(8, 500, 300)]
    sample = build_training_sample(
        previous,
        current,
        points_previous,
        points_current,
        np.random.default_rng(0),
    )
    assert sample.inputs.shape == (3, 544, 960)
    assert sample.center.shape == (1, 136, 240)
    assert sample.displacement.shape == (2, 136, 240)
    assert sample.displacement_mask.sum() == 1
    assert sample.inputs[0].max() == 1
    assert sample.inputs[2].max() == 1


def test_inference_input_feedback_uses_previous_track_positions():
    image = np.zeros((32, 48), dtype=np.uint8)
    inputs = make_input(image, image, [TrackPoint(2, 12, 10, 0.9)])
    assert inputs.shape == (3, 32, 48)
    assert inputs[2, 10, 12] == 1


def test_motion_target_survives_prior_crossing_crop_boundary():
    image = np.zeros((600, 1000), dtype=np.uint8)
    sample = build_training_sample(
        image,
        image,
        [TrackedPoint(3, -5, 300)],
        [TrackedPoint(3, 5, 300)],
        np.random.default_rng(0),
    )
    assert sample.origin_xy == (0, 28)
    assert sample.displacement_mask.sum() == 1
