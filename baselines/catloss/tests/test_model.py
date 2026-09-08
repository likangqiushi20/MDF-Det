import unittest

import torch

from catloss.heatmaps import binary_center_map, extract_peaks
from catloss.model import LocalizationNetwork, ObjectnessNetwork


class CatLossModelTests(unittest.TestCase):
    def test_objectness_shape(self) -> None:
        model = ObjectnessNetwork()
        model.eval()
        self.assertEqual(tuple(model(torch.zeros(2, 4, 21, 21)).shape), (2, 2))

    def test_localization_shapes(self) -> None:
        for mode in ("half", "full"):
            model = LocalizationNetwork(dilation_mode=mode)
            model.eval()
            output = model(torch.zeros(2, 4, 45, 45))
            self.assertEqual(tuple(output.shape), (2, 1, 15, 15))

    def test_binary_centers_and_peaks(self) -> None:
        target = binary_center_map(15, 15, [(3.1, 4.2), (10.0, 11.0)])
        peaks = extract_peaks(target[None], threshold=0.5)
        self.assertEqual({peak[:2] for peak in peaks[0]}, {(3, 4), (10, 11)})


if __name__ == "__main__":
    unittest.main()
