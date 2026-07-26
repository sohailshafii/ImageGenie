"""The multi-view CNN classifier (M6 B5, ml.md#representation / #training).

A shared 2D backbone runs on each rendered view, the per-view features are pooled
across the views, and a small classifier head maps the pooled feature to the 12
classes. Everything is config-driven (ml/train.py's ``Config``) — backbone,
pooling, head shape, dropout, and class count all come from the run's config,
which is recorded for reproducibility (NFR-4).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

# backbone name -> (constructor, ImageNet weights enum). Each exposes ``.fc``,
# which is swapped for Identity so the backbone emits features, not ImageNet logits.
_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name not in _BACKBONES:
        raise ValueError(f"unsupported backbone {name!r}; known: {sorted(_BACKBONES)}")
    constructor, weights = _BACKBONES[name]
    backbone = constructor(weights=weights if pretrained else None)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()  # emit features instead of ImageNet logits
    return backbone, feature_dim


def _build_head(
    feature_dim: int, hidden_dims: list[int], num_classes: int, dropout: float
) -> nn.Sequential:
    layers: list[nn.Module] = []
    input_dim = feature_dim
    for hidden_dim in hidden_dims:
        layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, num_classes))
    return nn.Sequential(*layers)


class MultiViewCNN(nn.Module):
    """Shared backbone over each view → pool across views → classifier head."""

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        view_pool: str,
        head_hidden_dims: list[int],
        dropout: float,
        num_classes: int,
        feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        if view_pool not in ("max", "mean"):
            raise ValueError(f"unsupported view_pool {view_pool!r}; use 'max' or 'mean'")
        self._backbone, backbone_dim = _build_backbone(backbone, pretrained)
        # feature_dim is a declared config knob; validate it against the real
        # backbone so a stale value (e.g. 512 left on a resnet50) fails loudly.
        if feature_dim is not None and feature_dim != backbone_dim:
            raise ValueError(
                f"config feature_dim {feature_dim} != {backbone} output {backbone_dim}"
            )
        self._view_pool = view_pool
        self._head = _build_head(backbone_dim, head_hidden_dims, num_classes, dropout)

    @classmethod
    def from_config(cls, config) -> MultiViewCNN:
        """Build from a run ``Config`` (ml/train.py). Duck-typed rather than
        importing ``Config`` here, since train.py imports this module."""
        return cls(
            backbone=config.backbone,
            pretrained=config.pretrained,
            view_pool=config.view_pool,
            head_hidden_dims=config.head_hidden_dims,
            dropout=config.dropout,
            num_classes=config.num_classes,
            feature_dim=config.feature_dim,
        )

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        # views: [batch, num_views, 3, H, W]. Fold the views into the batch so the
        # shared backbone sees every view, then regroup to pool per model.
        batch_size, num_views = views.shape[:2]
        flattened = views.flatten(0, 1)  # [batch*num_views, 3, H, W]
        features = self._backbone(flattened)  # [batch*num_views, feature_dim]
        features = features.view(batch_size, num_views, -1)  # [batch, num_views, F]
        pooled = (
            features.amax(dim=1) if self._view_pool == "max" else features.mean(dim=1)
        )  # [batch, F]
        return self._head(pooled)  # [batch, num_classes]
