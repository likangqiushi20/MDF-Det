import numpy as np
import torch

from hm_net.data import build_training_sample
from hm_net.infer import _pad_inputs
from hm_net.targets import ClassifiedPoint


def test_hmnet_paper_crop_and_three_input_branches():
    image = np.zeros((600, 1000), dtype=np.uint8)
    previous = [ClassifiedPoint(4, 496.25, 304.5, 0)]
    current = [ClassifiedPoint(4, 500.5, 300.25, 0)]
    sample = build_training_sample(
        image, image, previous, current, np.random.default_rng(3)
    )
    assert sample.current.shape == (1, 544, 960)
    assert sample.previous.shape == (1, 544, 960)
    assert sample.feedback.shape == (2, 544, 960)
    assert sample.center.shape == (2, 544, 960)
    assert sample.motion_mask.sum() == 1
    assert sample.precision_mask.sum() == 1
    assert sample.feedback.max() <= 1


def test_full_aoi_inputs_pad_to_sixteen():
    current = torch.zeros(1, 1, 1400, 2000)
    previous = torch.zeros_like(current)
    feedback = torch.zeros(1, 2, 1400, 2000)
    padded = _pad_inputs(current, previous, feedback)
    assert [tensor.shape[-2:] for tensor in padded] == [
        (1408, 2000),
        (1408, 2000),
        (1408, 2000),
    ]
