import unittest

import torch

from catloss.losses import CrowdAwareThresholdedLoss, ThresholdedLoss


class CatLossTests(unittest.TestCase):
    def test_background_below_tau_is_free(self) -> None:
        predictions = torch.full((1, 1, 2, 2), 0.2)
        targets = torch.zeros_like(predictions)
        self.assertEqual(ThresholdedLoss(tau=0.2)(predictions, targets).item(), 0.0)

    def test_single_target_catloss_equals_tloss(self) -> None:
        targets = torch.zeros((1, 1, 2, 2))
        targets[0, 0, 0, 0] = 1
        predictions = torch.zeros_like(targets)
        tloss = ThresholdedLoss()(predictions, targets)
        catloss = CrowdAwareThresholdedLoss(q=0.7)(predictions, targets)
        self.assertAlmostEqual(tloss.item(), catloss.item())

    def test_four_targets_q_half_doubles_loss(self) -> None:
        targets = torch.ones((1, 1, 2, 2))
        predictions = torch.zeros_like(targets)
        tloss = ThresholdedLoss()(predictions, targets)
        catloss = CrowdAwareThresholdedLoss(q=0.5)(predictions, targets)
        self.assertAlmostEqual(catloss.item(), 2 * tloss.item())

    def test_empty_localization_patch_is_rejected(self) -> None:
        values = torch.zeros((1, 1, 2, 2))
        with self.assertRaises(ValueError):
            CrowdAwareThresholdedLoss()(values, values)
