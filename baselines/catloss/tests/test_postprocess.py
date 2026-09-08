import numpy as np

from catloss.postprocess import extract_foreground_regions


def test_foreground_region_filters_tiny_component():
    images = [np.zeros((31, 31), dtype=np.uint8) for _ in range(3)]
    current = np.zeros((31, 31), dtype=np.uint8)
    current[10:13, 10:13] = 255
    current[20, 20] = 255
    images.append(current)
    mask, regions = extract_foreground_regions(
        images,
        foreground_quantile=0.5,
    )
    assert len(regions) == 1
    assert regions[0].area == 9
    assert mask.sum() == 9


def test_empty_temporal_change_returns_empty_regions():
    images = [np.zeros((9, 9), dtype=np.uint8) for _ in range(4)]
    mask, regions = extract_foreground_regions(images)
    assert not regions
    assert mask.sum() == 0
