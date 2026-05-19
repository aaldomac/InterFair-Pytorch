from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

@dataclass
class DFRegularizerConfig:
    enabled: bool = True
    lambda_df: float = 1e-2
    alpha: float = 1.0
    epsilon_target: float = 0.0
    burn_in_epochs: int = 10
    fairness_source: str = "val"  # "train" or "val"
    eps: float = 1e-12


def soft_group_positive_rates(
    probs_positive: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    probs_positive = probs_positive.view(-1)
    group_ids = group_ids.view(-1).long()

    n_groups = int(group_ids.max().item()) + 1

    counts = torch.bincount(group_ids, minlength=n_groups).to(
        device=probs_positive.device,
        dtype=probs_positive.dtype,
    )

    positive_mass = torch.zeros(
        n_groups,
        device=probs_positive.device,
        dtype=probs_positive.dtype,
    )
    positive_mass.scatter_add_(0, group_ids, probs_positive)

    # Binary outcomes: denominator N_s + 2 * alpha
    p_pos = (positive_mass + alpha) / (counts + 2.0 * alpha)
    return p_pos


def differential_fairness_epsilon_soft(
    probs_positive: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    alpha: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    p_pos = soft_group_positive_rates(probs_positive, group_ids, alpha=alpha)
    p_pos = p_pos.clamp(min=eps, max=1.0 - eps)
    p_neg = 1.0 - p_pos

    log_pos = torch.log(p_pos)
    log_neg = torch.log(p_neg)

    eps_pos = torch.abs(log_pos[:, None] - log_pos[None, :])
    eps_neg = torch.abs(log_neg[:, None] - log_neg[None, :])

    return torch.maximum(eps_pos, eps_neg).max()


class DifferentialFairnessRegularizer(nn.Module):
    def __init__(self, cfg: DFRegularizerConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        *,
        model: nn.Module,
        fair_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.cfg.enabled or epoch < self.cfg.burn_in_epochs:
            zero = torch.tensor(0.0, device=device)
            return zero, {"df_epsilon": 0.0, "df_penalty": 0.0}

        was_training = model.training
        model.eval()

        probs_all = []
        groups_all = []

        for x_fair, _, g_fair in fair_loader:
            x_fair = x_fair.to(device=device, dtype=torch.float32)
            g_fair = g_fair.to(device=device, dtype=torch.long).view(-1)

            probs = model(x_fair).squeeze(-1)
            probs_all.append(probs)
            groups_all.append(g_fair)

        probs_all = torch.cat(probs_all, dim=0)
        groups_all = torch.cat(groups_all, dim=0)

        epsilon_df = differential_fairness_epsilon_soft(
            probs_all,
            groups_all,
            alpha=self.cfg.alpha,
            eps=self.cfg.eps,
        )

        penalty = self.cfg.lambda_df * torch.relu(
            epsilon_df - torch.tensor(self.cfg.epsilon_target, device=device, dtype=epsilon_df.dtype)
        )

        if was_training:
            model.train()

        return penalty, {
            "df_epsilon": float(epsilon_df.item()),
            "df_penalty": float(penalty.item()),
        }