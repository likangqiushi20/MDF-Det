import numpy as np
import pytest
import torch

from hm_net.losses import hmnet_loss
from hm_net.model import Decoder, Encoder, HMNet
from hm_net.sgr import (
    augment_feedback_centers,
    selective_gaussian_reconstruction,
)
from hm_net.targets import ClassifiedPoint, build_targets
from hm_net.tracking import Detection, Track, associate_hungarian, decode


def test_hmnet_heads_preserve_input_resolution():
    model = HMNet().eval()
    with torch.no_grad():
        output = model(
            torch.zeros(1, 1, 64, 96),
            torch.zeros(1, 1, 64, 96),
            torch.zeros(1, 2, 64, 96),
        )
    assert output["center"].shape == (1, 2, 64, 96)
    assert output["motion"].shape == (1, 2, 64, 96)
    assert output["precision"].shape == (1, 2, 64, 96)


def test_pre_pooling_skips_align_with_every_decoder_scale():
    encoder = Encoder(1).eval()
    decoder = Decoder().eval()
    with torch.no_grad():
        bottleneck, skips = encoder(torch.zeros(1, 1, 64, 96))
        _, decoder_outputs = decoder(bottleneck, skips)
    assert [value.shape[-2:] for value in skips] == [
        (64, 96), (32, 48), (16, 24), (8, 12)
    ]
    assert [value.shape for value in decoder_outputs] == [
        value.shape for value in reversed(skips)
    ]


def test_targets_encode_class_subpixel_and_motion():
    current = [ClassifiedPoint(7, 10.25, 20.75, 0)]
    previous = [ClassifiedPoint(7, 8.25, 24.75, 0)]
    center, motion, precision, motion_mask, precision_mask = build_targets(
        current, previous, 32, 40
    )
    assert center.shape == (2, 32, 40)
    assert center[0, 20, 10] == 1
    assert motion[:, 20, 10].tolist() == pytest.approx([-2, 4])
    assert precision[:, 20, 10].tolist() == pytest.approx([0.25, 0.75])
    assert motion_mask.sum() == 1
    assert precision_mask.sum() == 1


def test_combined_loss_is_finite():
    center, motion, precision, motion_mask, precision_mask = build_targets(
        [ClassifiedPoint(1, 10.5, 12.25, 0)], [], 32, 40
    )
    predictions = {
        "center": torch.full((1, 2, 32, 40), 0.1),
        "motion": torch.zeros(1, 2, 32, 40),
        "precision": torch.full((1, 2, 32, 40), 0.5),
    }
    total, parts = hmnet_loss(
        predictions,
        center[None],
        motion[None],
        precision[None],
        motion_mask[None],
        precision_mask[None],
    )
    assert torch.isfinite(total)
    assert set(parts) == {"center", "motion", "precision"}


def test_sgr_filters_weak_peak_and_amplifies_detection():
    heatmap = torch.zeros(1, 2, 32, 32)
    heatmap[0, 0, 8, 8] = 0.2
    heatmap[0, 1, 20, 20] = 0.5
    reconstructed = selective_gaussian_reconstruction(heatmap)
    assert reconstructed[0, 0].max() == 0
    assert reconstructed[0, 1].max() == pytest.approx(0.6)


def test_rcr_rcp_adds_one_false_center_per_truth():
    augmented = augment_feedback_centers(
        [(5, 6, 0)], np.random.default_rng(2)
    )
    assert len(augmented) == 2
    assert 0.2 <= augmented[0][3] <= 1


def test_decode_precision_and_hungarian_identity():
    center = torch.zeros(1, 2, 32, 32)
    motion = torch.zeros(1, 2, 32, 32)
    precision = torch.zeros(1, 2, 32, 32)
    center[0, 0, 10, 12] = 0.9
    motion[0, :, 10, 12] = torch.tensor([-2.0, 3.0])
    precision[0, :, 10, 12] = torch.tensor([0.25, 0.75])
    detections = decode(center, motion, precision)[0]
    assert (detections[0].x, detections[0].y) == (12.25, 10.75)
    previous = [Track(4, 10.25, 13.75, 0, 0.8)]
    tracks, next_id = associate_hungarian(detections, previous, 9)
    assert tracks[0].track_id == 4
    assert next_id == 9


def test_decode_caps_flat_low_confidence_plateaus():
    center = torch.full((1, 2, 32, 32), 0.1)
    motion = torch.zeros(1, 2, 32, 32)
    precision = torch.zeros(1, 2, 32, 32)
    detections = decode(
        center,
        motion,
        precision,
        threshold=0.05,
        max_detections_per_class=3,
    )[0]
    assert len(detections) == 6


def test_sgr_threshold_tracks_a_lower_calibrated_detection_threshold():
    heatmap = torch.zeros(1, 2, 32, 32)
    heatmap[0, 0, 8, 8] = 0.1
    reconstructed = selective_gaussian_reconstruction(
        heatmap,
        filter_threshold=0.28,
        detection_threshold=0.05,
    )
    assert reconstructed[0, 0].max() > 0
