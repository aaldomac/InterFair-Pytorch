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

    if probs_positive.numel() == 0 or probs_positive.shape != group_ids.shape:
        raise ValueError('Require nonempty, aligned probabilities and group IDs')
    if not torch.isfinite(probs_positive).all() or (probs_positive < 0).any() or (probs_positive > 1).any():
        raise ValueError('Expected finite probabilities in [0, 1]')
    if not torch.isfinite(torch.tensor(alpha)) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    _, group_ids = torch.unique(group_ids.to(probs_positive.device), sorted=True, return_inverse=True)
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
    if not 0 < eps < .5:
        raise ValueError('eps must lie in (0, .5)')
    clamp_eps = max(eps, torch.finfo(p_pos.dtype).eps)
    p_pos = p_pos.clamp(min=clamp_eps, max=1.0-clamp_eps)
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

    def forward(self, context=None, *, model=None, fair_loader=None, device=None, epoch=None):
        """Accept RegularizerContext and retain the legacy keyword-call interface.

        lambda_df is this penalty's internal weight. CompositeLoss can apply a
        further regularizer_weight; leave one at 1 to avoid double weighting.
        """
        from modules.predictive.metrics import logits_to_probs
        if context is not None:
            model, fair_loader, device, epoch = (context.model, context.fair_loader,
                                                context.device, context.epoch)
        if model is None or device is None or epoch is None:
            raise ValueError('Pass RegularizerContext or model, device and epoch')
        if not self.cfg.enabled or epoch < self.cfg.burn_in_epochs:
            return torch.zeros((), device=device), {'df_epsilon': 0., 'df_penalty': 0.}
        if fair_loader is None:
            raise ValueError('An explicit fairness loader is required')
        was_training = model.training
        model.eval()
        probs_all, groups_all = [], []
        try:
            for x_fair, _, g_fair in fair_loader:
                logits = model(x_fair.to(device=device, dtype=torch.float32))
                binary_logits = logits.ndim == 1 or logits.shape[-1] == 1
                probs = logits_to_probs(logits, binary=binary_logits)
                if probs.shape[1] != 2:
                    raise ValueError('DifferentialFairnessRegularizer currently supports binary outcomes')
                probs_all.append(probs[:, 1])
                groups_all.append(g_fair.to(device=device, dtype=torch.long).view(-1))
            if not probs_all:
                raise ValueError('Fairness loader is empty')
            epsilon_df = differential_fairness_epsilon_soft(torch.cat(probs_all),
                torch.cat(groups_all), alpha=self.cfg.alpha, eps=self.cfg.eps)
            penalty = self.cfg.lambda_df * torch.relu(epsilon_df-self.cfg.epsilon_target)
            return penalty, {'df_epsilon': float(epsilon_df.detach().item()),
                             'df_penalty': float(penalty.detach().item())}
        finally:
            model.train(was_training)