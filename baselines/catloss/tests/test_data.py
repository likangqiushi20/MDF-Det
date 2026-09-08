import numpy as np
import pytest

from catloss.data import (
    augment_patch_batches,
    build_localization_batch,
    build_patch_pools,
    crop_stack,
    sample_objectness_epoch,
    temporal_change_score,
)


def test_crop_stack_normalizes_uint8_and_keeps_temporal_order():
    images = [np.full((9, 9), value, dtype=np.uint8) for value in (0, 255)]
    patch = crop_stack(images, 4, 4, 5)
    assert patch.shape == (2, 5, 5)
    assert patch[0].max() == 0.0
    assert patch[1].min() == 1.0


def test_temporal_change_uses_past_only_for_background():
    images = [np.zeros((9, 9), dtype=np.uint8) for _ in range(3)]
    images.append(np.zeros((9, 9), dtype=np.uint8))
    images[-1][4, 4] = 20
    score = temporal_change_score(images)
    assert score[4, 4] == 20
    assert np.count_nonzero(score) == 1


def test_pools_exclude_truth_and_epoch_sampler_balances_classes():
    images = [np.zeros((31, 31), dtype=np.uint8) for _ in range(3)]
    images.append(np.zeros((31, 31), dtype=np.uint8))
    images[-1][20, 20] = 255
    pools = build_patch_pools(
        images,
        [(15.0, 15.0)],
        patch_size=5,
        ordinary_stride=5,
        hard_stride=1,
        hard_quantile=0.99,
        exclusion_radius=3,
    )
    assert pools.positive == ((15, 15),)
    assert (20, 20) in pools.hard_negative
    inputs, targets = sample_objectness_epoch(
        images,
        pools,
        np.random.default_rng(4),
        patch_size=5,
    )
    assert inputs.shape == (2, 4, 5, 5)
    assert sorted(targets.tolist()) == [0, 1]


def test_localization_batch_contains_all_centers_in_patch():
    images = [np.zeros((51, 51), dtype=np.uint8) for _ in range(4)]
    inputs, targets = build_localization_batch(
        images,
        [(25.0, 25.0), (28.0, 25.0)],
        [(25, 25)],
        patch_size=45,
        output_size=15,
    )
    assert inputs.shape == (1, 4, 45, 45)
    assert targets.shape == (1, 1, 15, 15)
    assert targets.sum() == 2


def test_even_patch_is_rejected():
    with pytest.raises(ValueError):
        crop_stack([np.zeros((9, 9))], 4, 4, 4)


def test_augmentation_translates_local_input_and_target_together():
    object_x = np.ones((1, 4, 5, 5), dtype=np.float32)
    local_x = np.zeros((1, 4, 9, 9), dtype=np.float32)
    local_y = np.zeros((1, 1, 3, 3), dtype=np.float32)
    local_x[:, :, 3:6, 3:6] = 1
    local_y[:, :, 1, 1] = 1
    _, shifted_x, shifted_y = augment_patch_batches(
        object_x,
        local_x,
        local_y,
        np.random.default_rng(8),
        intensity_range=(1.0, 1.0),
        object_shift_px=0,
        local_output_shift=1,
    )
    target_row, target_col = np.argwhere(shifted_y[0, 0] == 1)[0]
    input_center = np.argwhere(shifted_x[0, 0] == 1).mean(axis=0)
    assert np.allclose(input_center, [3 * target_row + 1, 3 * target_col + 1])
