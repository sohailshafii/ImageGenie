"""MultiViewCNN forward + config wiring (M6 B5)."""

import pytest
import torch
from model import MultiViewCNN
from train import Config


def _tiny_model(**overrides) -> MultiViewCNN:
    # pretrained=False so no ImageNet weights are downloaded during the test run.
    config = Config(pretrained=False, **overrides)
    return MultiViewCNN.from_config(config)


def test_forward_maps_views_to_class_logits() -> None:
    model = _tiny_model()
    model.eval()
    # [batch=2, num_views=3, 3, 64, 64] — small views keep the CPU forward fast.
    logits = model(torch.zeros(2, 3, 3, 64, 64))
    assert logits.shape == (2, 12)  # Config.num_classes


def test_mean_pool_is_supported() -> None:
    model = _tiny_model(view_pool="mean")
    logits = model(torch.zeros(1, 4, 3, 64, 64))
    assert logits.shape == (1, 12)


def test_unknown_view_pool_is_rejected() -> None:
    with pytest.raises(ValueError):
        _tiny_model(view_pool="median")


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(ValueError):
        _tiny_model(backbone="vgg16")


def test_feature_dim_mismatch_is_rejected() -> None:
    # resnet18 emits 512 features; declaring 2048 must fail loudly, not mis-size.
    with pytest.raises(ValueError):
        _tiny_model(feature_dim=2048)
