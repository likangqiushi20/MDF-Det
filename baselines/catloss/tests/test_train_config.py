import pytest
from catloss.losses import CrowdAwareThresholdedLoss, ThresholdedLoss
from catloss.train import make_localization_loss


def test_localization_ablation_loss_factory():
    assert make_localization_loss({"localization_loss": "mse"}).__class__.__name__ == "MSELoss"
    assert isinstance(
        make_localization_loss({"localization_loss": "tloss", "tau": 0.2}),
        ThresholdedLoss,
    )
    assert isinstance(
        make_localization_loss(
            {"localization_loss": "catloss", "tau": 0.2, "q": 0.5}
        ),
        CrowdAwareThresholdedLoss,
    )


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError):
        make_localization_loss({"localization_loss": "unknown"})
