from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

import torch
import torch.nn as nn

class Regularizer(Protocol):
    def __call__(self, context: "RegularizerContext") -> tuple[torch.Tensor, dict[str, float]]:
        ...

@dataclass(frozen=True)
class RegularizerContext:
    """
    Context object passed to regularizer functions, containing all necessary information to compute the regularization term.
    """
    model: nn.Module
    device: torch.device
    epoch: int
    global_step: int
    fair_loader: Optional[torch.utils.data.DataLoader] = None
    extras: Mapping[str, Any] = field(default_factory=dict)

class NoRegularizer(nn.Module):
    def forward(self, context: RegularizerContext) -> tuple[torch.Tensor, dict[str, float]]:
        return torch.zeros((), devce=context.device), {}
    
class CompositeLoss(nn.Module):
    """
    Combines supervised task loss with an optional regularizer.
    """
    def __init__(
        self, 
        task_loss: nn.Module,
        regularizer: Optional[Regularizer] = None,
        regularizer_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if regularizer_weight < 0.0:
            raise ValueError("regularizer_weight must be non-negative.")
        self.task_loss = task_loss
        self.regularizer = regularizer
        self.regularizer_weight = regularizer_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        context: RegularizerContext,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        task = self.task_loss(logits, targets)

        if self.regularizer is None:
            reg = torch.zeros((), device=context.device)
            reg_stats: dict[str, float] = {}
        else:
            reg, reg_stats = self.regularizer(context)

        total = task + self.regularizer_weight * reg
        stats = {
            "loss": float(total.detach().item()),
            "task_loss": float(task.detach().item()),
            "reg_loss": float(reg.detach().item()),
            **reg_stats,
        }
        return total, stats
    
class MulticlassClassificationLoss(nn.Module):
    """
    Cross-entropy loss for multi-class classification.

    Expects:
    - `logits` of shape `(batch_size, num_classes)`
    - `targets` of shape `(batch_size,)` with integer class labels in `[0, num_classes-1]`
    """
    def __init__(self, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets.long().view(-1))
    