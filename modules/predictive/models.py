from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    """
    A simple multi-layer perceptron (MLP) classifier returning logits.

    Args:
        input_dim: The number of input features.
        num_outputs: The number of output classes. Use 2 for binary classification if explicit two-logit output. Use 1 for binary BCE-style training.
        hidden_dims: A sequence of integers specifying the number of units in each hidden layer.
        dropout: Dropout probability.
        activation: Activation function to use between layers. Default is ReLU.
    """

    def __init__(
        self,
        input_dim: int,
        num_outputs: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive.")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size.")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError("All hidden layer sizes must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        
        self.input_dim = input_dim
        self.num_outputs = num_outputs

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_outputs))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected input of shape (batch_size, input_dim), got {tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input with {self.input_dim} features, got {x.shape[1]}")
        return self.model(x)

class ImageClassifier(nn.Module):
    """Compact CNN baseline with adaptive pooling for CelebA images."""
    def __init__(self, num_outputs=1, dropout=0.2):
        super().__init__()
        self.num_outputs = num_outputs
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(128, num_outputs))

    def forward(self, x):
        return self.net(x)
