"""Vision-to-language projector. Imports torch, so only import this lazily."""

from __future__ import annotations

import torch.nn as nn


class ProjectionMLP(nn.Module):
    """SigLIP2 patch features -> N-ATLaS embedding space (the part trained in stage 1)."""

    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(vision_dim, text_dim), nn.GELU(), nn.Linear(text_dim, text_dim))

    def forward(self, x):
        return self.net(x)
