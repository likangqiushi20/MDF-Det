import numpy as np

from catloss.data import build_patch_pools, sample_objectness_epoch


def test_fractional_truth_rounding_cannot_create_invalid_patch_center():
    images = [np.zeros((100, 100), dtype=np.uint8) for _ in range(4)]
    pools = build_patch_pools(
        images,
        [(89.6, 50.0), (50.0, 50.0)],
        hard_quantile=0.99,
    )
    assert (90, 50) not in pools.positive
    inputs, targets = sample_objectness_epoch(
        images, pools, np.random.default_rng(1), negatives_per_positive=1
    )
    assert inputs.shape[-2:] == (21, 21)
    assert len(inputs) == len(targets) == 2
