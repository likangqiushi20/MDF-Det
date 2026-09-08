import pytest
import torch

from hmrn.losses import CenterFocalLoss, masked_displacement_l1
from hmrn.targets import TrackedPoint, build_targets, gaussian_heatmap


def test_gaussian_heatmap_has_unit_discrete_peak():
    heatmap = gaussian_heatmap(16, 20, [(7, 5)], sigma=1.5)
    assert heatmap.shape == (16, 20)
    assert heatmap[5, 7] == 1


def test_targets_encode_previous_minus_current_motion():
    current = [TrackedPoint(11, 28, 20), TrackedPoint(12, 60, 40)]
    previous = [TrackedPoint(11, 24, 28)]
    heatmap, displacement, mask = build_targets(
        current, previous, input_height=64, input_width=80
    )
    assert heatmap.shape == (1, 16, 20)
    assert displacement.shape == (2, 16, 20)
    assert mask.shape == (1, 16, 20)
    assert heatmap[0, 5, 7] == 1
    assert displacement[:, 5, 7].tolist() == pytest.approx([-1, 2])
    assert mask.sum() == 1


def test_focal_loss_rewards_correct_center_prediction():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 3, 4] = 1
    good = torch.full_like(target, 0.01)
    bad = torch.full_like(target, 0.01)
    good[0, 0, 3, 4] = 0.99
    bad[0, 0, 3, 4] = 0.01
    loss = CenterFocalLoss()
    assert loss(good, target) < loss(bad, target)


def test_masked_l1_ignores_background():
    prediction = torch.full((1, 2, 4, 4), 100.0)
    target = torch.zeros_like(prediction)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[0, 0, 2, 1] = True
    prediction[0, :, 2, 1] = torch.tensor([2.0, -4.0])
    assert masked_displacement_l1(prediction, target, mask) == 3
