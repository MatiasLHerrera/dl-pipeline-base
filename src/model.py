"""Arquitectura base: MLP construido con nn.Sequential dentro de un nn.Module."""
import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Perceptrón multicapa simple: [Linear -> ReLU -> Dropout] x N -> Linear (logits)."""

    def __init__(self, in_dim: int, n_classes: int, hidden_dims=(16, 16), dropout: float = 0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))  # sin softmax: CrossEntropyLoss lo aplica internamente
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
