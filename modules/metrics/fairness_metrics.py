from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[int]]

from modules.metrics.performance_metrics import (
    true_positive_rate,
    false_positive_rate,
    prediction_scores_to_labels,
    positive_class_scores,
)
from modules.utils.dataset_utils import compute_pg_dirichlet_from_groups
from modules.utils.tensor_utils import (
    _as_tensor,
    _as_1d_tensor,
    _validate_same_length,
    _unique_sorted_long,
)


# ==================================================
# Result containers
# ==================================================
@dataclass
class PairwiseFairnessResult:
    """Container for group fairness metrics."""

    aggregate: float
    matrix: torch.Tensor
    groups: torch.Tensor
    per_group_values: Optional[torch.Tensor] = None


@dataclass
class SubgroupFairnessResult:
    """Container for subgroup-vs-population fairness metrics."""

    aggregate: float
    values: torch.Tensor
    groups: torch.Tensor
    global_value: float
    group_values: torch.Tensor
    group_probs: torch.Tensor


# ==================================================
# Generic helpers
# ==================================================
def _build_pairwise_matrix(
    unique_groups: torch.Tensor,
    pairwise_fn,
    diagonal_value: float,
) -> torch.Tensor:
    """
    Build a symmetric pairwise matrix M where M[i, j] is the pairwise metric
    between group unique_groups[i] and group unique_groups[j].

    pairwise_fn(i, j) must return a scalar.
    """
    n_groups = len(unique_groups)
    matrix = torch.full((n_groups, n_groups), float(diagonal_value), dtype=torch.float32)

    for i, j in itertools.combinations(range(n_groups), 2):
        value = float(pairwise_fn(i, j))
        matrix[i, j] = value
        matrix[j, i] = value

    return matrix


def _max_off_diagonal(matrix: torch.Tensor) -> float:
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    return float(matrix[mask].max().item())


def _min_off_diagonal(matrix: torch.Tensor) -> float:
    n = matrix.shape[0]
    if n < 2:
        return 1.0
    mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    return float(matrix[mask].min().item())


def _positive_prediction_rate(preds: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Positive prediction rate for binary hard labels or binary scores.
    P(Y_hat=1) or P(score >= threshold).
    """
    preds = preds.view(-1)
    if preds.numel() == 0:
        return 0.0
    return float((preds >= threshold).float().mean().item())

# TODO: Think if this function makes sense and will be used
def _class_prediction_rate(pred_labels: torch.Tensor, positive_class: int = 1) -> float:
    pred_labels = pred_labels.view(-1).long()
    if pred_labels.numel() == 0:
        return 0.0
    return float((pred_labels == positive_class).float().mean().item())


def _positive_scores(
    predictions_or_probs: TensorLike,
    *,
    binary: bool = True,
    threshold: float = 0.5,
    positive_class: int = 1,
) -> torch.Tensor:
    """
    Return binary positive-class scores or multiclass one-vs-rest hard indicators.
    Number of correct predictions.
    """
    x = _as_tensor(predictions_or_probs)

    if binary:
        if x.ndim == 2 and x.shape[1] == 2:
            return x[:, positive_class].float().view(-1)
        if x.ndim == 2 and x.shape[1] == 1:
            return x.squeeze(1).float().view(-1)
        if x.ndim == 1:
            return x.float().view(-1)
        raise ValueError(f"Unsupported binary prediction shape: {tuple(x.shape)}")

    labels = prediction_scores_to_labels(
        x,
        binary=False,
        threshold=threshold,
        positive_class=positive_class,
    )
    return (labels == positive_class).float()

# TODO: Consider if this function makes sense and will be used
def _rate_for_positive_class(
    predictions_or_probs: TensorLike,
    *,
    binary: bool = True,
    threshold: float = 0.5,
    positive_class: int = 1,
) -> float:
    scores = _positive_scores(
        predictions_or_probs,
        binary=binary,
        threshold=threshold,
        positive_class=positive_class,
    )
    return _positive_prediction_rate(scores, threshold=threshold if binary else 0.5)


# ==================================================
# Pairwise prediction metrics: two groups
# ==================================================

def statistical_parity_two_groups(
    preds_a: torch.Tensor,
    preds_b: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """| P(predicted positive | A) - P(predicted positive | B) |."""
    rate_a = _positive_prediction_rate(preds_a, threshold)
    rate_b = _positive_prediction_rate(preds_b, threshold)
    return abs(rate_a - rate_b)


def equal_opportunity_two_groups(
    preds_a: torch.Tensor,
    labels_a: torch.Tensor,
    preds_b: torch.Tensor,
    labels_b: torch.Tensor,
    *,
    threshold: float = 0.5,
    positive_class: int = 1,
) -> float:
    """| TPR(A) - TPR(B) |."""
    tpr_a = true_positive_rate(
        preds_a,
        labels_a,
        threshold=threshold,
        positive_class=positive_class,
    )
    tpr_b = true_positive_rate(
        preds_b,
        labels_b,
        threshold=threshold,
        positive_class=positive_class,
    )
    return abs(tpr_a - tpr_b)


def disparate_impact_two_groups(
    preds_a: torch.Tensor,
    preds_b: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Symmetric disparate impact ratio:
        min(rate_a, rate_b) / max(rate_a, rate_b)

    Range: [0, 1]. Best value: 1.
    """
    rate_a = _positive_prediction_rate(preds_a, threshold)
    rate_b = _positive_prediction_rate(preds_b, threshold)

    max_rate = max(rate_a, rate_b)
    min_rate = min(rate_a, rate_b)

    if max_rate == 0:
        return 1.0

    return min_rate / max_rate


def equalized_odds_two_groups(
    preds_a: torch.Tensor,
    labels_a: torch.Tensor,
    preds_b: torch.Tensor,
    labels_b: torch.Tensor,
    *,
    threshold: float = 0.5,
    positive_class: int = 1,
) -> float:
    """max(|TPR(A)-TPR(B)|, |FPR(A)-FPR(B)|)."""
    tpr_a = true_positive_rate(
        preds_a,
        labels_a,
        threshold=threshold,
        positive_class=positive_class,
    )
    tpr_b = true_positive_rate(
        preds_b,
        labels_b,
        threshold=threshold,
        positive_class=positive_class,
    )

    fpr_a = false_positive_rate(
        preds_a,
        labels_a,
        threshold=threshold,
        positive_class=positive_class,
    )
    fpr_b = false_positive_rate(
        preds_b,
        labels_b,
        threshold=threshold,
        positive_class=positive_class,
    )

    return max(abs(tpr_a - tpr_b), abs(fpr_a - fpr_b))


# ==================================================
# Full prediction metrics
# ==================================================

def statistical_parity(
    preds: TensorLike,
    group_ids: TensorLike,
    threshold: float = 0.5,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Returns:
        aggregate: max pairwise statistical parity gap
        matrix: [G, G] absolute pairwise gaps
        groups: group IDs in matrix order
    """
    preds_t = _as_1d_tensor(preds, dtype=torch.float32, name="preds")
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")
    _validate_same_length(preds=preds_t, group_ids=group_ids_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")

    if len(groups) < 2:
        matrix = torch.zeros((len(groups), len(groups)), dtype=torch.float32)
        return 0.0, matrix, groups

    def pairwise_fn(i: int, j: int) -> float:
        g1, g2 = groups[i], groups[j]
        preds_g1 = preds_t[group_ids_t == g1]
        preds_g2 = preds_t[group_ids_t == g2]
        return statistical_parity_two_groups(preds_g1, preds_g2, threshold)

    matrix = _build_pairwise_matrix(groups, pairwise_fn, diagonal_value=0.0)
    aggregate = _max_off_diagonal(matrix)

    return aggregate, matrix, groups


def equal_opportunity(
    preds: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    threshold: float = 0.5,
    positive_class: int = 1,
):
    preds_t = _as_1d_tensor(preds, dtype=torch.long, name="preds")
    labels_t = _as_1d_tensor(labels, dtype=torch.long, name="labels")
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")
    _validate_same_length(preds=preds_t, labels=labels_t, group_ids=group_ids_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")
    if len(groups) < 2:
        matrix = torch.zeros((len(groups), len(groups)), dtype=torch.float32)
        return 0.0, matrix, groups

    def pairwise_fn(i: int, j: int) -> float:
        g1, g2 = groups[i], groups[j]
        mask1 = group_ids_t == g1
        mask2 = group_ids_t == g2
        return equal_opportunity_two_groups(
            preds_t[mask1], labels_t[mask1],
            preds_t[mask2], labels_t[mask2],
            threshold=threshold,
            positive_class=positive_class,
        )

    matrix = _build_pairwise_matrix(groups, pairwise_fn, diagonal_value=0.0)
    aggregate = _max_off_diagonal(matrix)
    return aggregate, matrix, groups


def disparate_impact(
    preds: TensorLike,
    group_ids: TensorLike,
    threshold: float = 0.5,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Returns:
        aggregate: worst pairwise DI ratio = minimum off-diagonal entry
        matrix: [G, G] symmetric pairwise DI ratios
        groups: group IDs in matrix order
    """
    preds_t = _as_1d_tensor(preds, dtype=torch.float32, name="preds")
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")
    _validate_same_length(preds=preds_t, group_ids=group_ids_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")

    if len(groups) < 2:
        matrix = torch.ones((len(groups), len(groups)), dtype=torch.float32)
        return 1.0, matrix, groups

    def pairwise_fn(i: int, j: int) -> float:
        g1, g2 = groups[i], groups[j]
        preds_g1 = preds_t[group_ids_t == g1]
        preds_g2 = preds_t[group_ids_t == g2]
        return disparate_impact_two_groups(preds_g1, preds_g2, threshold)

    matrix = _build_pairwise_matrix(groups, pairwise_fn, diagonal_value=1.0)
    aggregate = _min_off_diagonal(matrix)

    return aggregate, matrix, groups


def equalized_odds(
    preds: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    threshold: float = 0.5,
    positive_class: int = 1,
):
    preds_t = _as_1d_tensor(preds, dtype=torch.long, name="preds")
    labels_t = _as_1d_tensor(labels, dtype=torch.long, name="labels")
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")
    _validate_same_length(preds=preds_t, labels=labels_t, group_ids=group_ids_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")

    if len(groups) < 2:
        matrix = torch.zeros((len(groups), len(groups)), dtype=torch.float32)
        return 0.0, matrix, groups

    def pairwise_fn(i: int, j: int) -> float:
        g1, g2 = groups[i], groups[j]
        mask1 = group_ids_t == g1
        mask2 = group_ids_t == g2
        return equalized_odds_two_groups(
            preds_t[mask1], labels_t[mask1],
            preds_t[mask2], labels_t[mask2],
            threshold=threshold,
            positive_class=positive_class,
        )

    matrix = _build_pairwise_matrix(groups, pairwise_fn, diagonal_value=0.0)
    aggregate = _max_off_diagonal(matrix)
    return aggregate, matrix, groups


# ==================================================
# Intersectional subgroup fairness
# ==================================================

def subgroup_statistical_parity_fairness_one_group(
    preds_group: torch.Tensor,
    preds_all: torch.Tensor,
    group_probability: float,
    threshold: float = 0.5,
) -> float:
    """
    Single-subgroup fairness violation:
        P(g) * | P(predicted positive) - P(predicted positive | g) |
    """
    global_rate = _positive_prediction_rate(preds_all, threshold)
    group_rate = _positive_prediction_rate(preds_group, threshold)
    return float(group_probability) * abs(global_rate - group_rate)


def subgroup_statistical_parity(
    preds: TensorLike,
    group_ids: TensorLike,
    threshold: float = 0.5,
    alpha: float = 1.0,
) -> Tuple[float, torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Group-vs-global subgroup statistical parity.

    Returns:
        aggregate: max subgroup fairness violation
        values: [G] tensor with P(g)*|P(predicted positive)-P(predicted positive|g)|
        groups: group IDs in the same order as values
        global_rate: P(predicted positive)
        group_rates: [G] tensor with P(predicted positive|g)
        group_probs: [G] tensor with smoothed P(g)
    """
    preds_t = _as_1d_tensor(preds, dtype=torch.float32, name="preds")
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")
    _validate_same_length(preds=preds_t, group_ids=group_ids_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")

    if len(groups) == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return 0.0, empty, groups, 0.0, empty, empty

    global_rate = _positive_prediction_rate(preds_t, threshold)

    # compute_pg_dirichlet_from_groups expects group IDs in [0, K-1].
    # To support arbitrary group labels, remap to contiguous IDs in the same order as `groups`.
    group_to_idx = {int(g.item()): i for i, g in enumerate(groups)}
    contiguous = torch.tensor(
        [group_to_idx[int(g.item())] for g in group_ids_t],
        dtype=torch.long,
        device=group_ids_t.device,
    )
    _, group_probs = compute_pg_dirichlet_from_groups(contiguous, K=len(groups), alpha=alpha)
    group_probs = group_probs.detach().cpu().float()

    values = []
    group_rates = []

    for idx, g in enumerate(groups):
        mask = group_ids_t == g
        preds_g = preds_t[mask]
        group_rate = _positive_prediction_rate(preds_g, threshold)
        group_rates.append(group_rate)
        values.append(
            subgroup_statistical_parity_fairness_one_group(
                preds_g,
                preds_t,
                group_probability=float(group_probs[idx].item()),
                threshold=threshold,
            )
        )

    values_t = torch.tensor(values, dtype=torch.float32)
    group_rates_t = torch.tensor(group_rates, dtype=torch.float32)
    aggregate = float(values_t.max().item()) if values_t.numel() > 0 else 0.0

    return aggregate, values_t, groups, global_rate, group_rates_t, group_probs


# ==================================================
# Differential fairness
# ==================================================

def _smoothed_positive_rate(
    preds: torch.Tensor,
    threshold: float = 0.5,
    alpha: float = 1.0,
) -> float:
    """
    Smoothed estimate of P(predicted positive) for binary predictions.

    p = (n_pos + alpha) / (n + 2*alpha)
    """
    preds = preds.view(-1)
    n = preds.numel()
    if n == 0:
        return 0.5

    n_pos = (preds >= threshold).float().sum().item()
    return float((n_pos + alpha) / (n + 2.0 * alpha))


def differential_fairness_two_groups(
    preds_a: torch.Tensor,
    preds_b: torch.Tensor,
    threshold: float = 0.5,
    alpha: float = 1.0,
) -> float:
    """
    Pairwise epsilon for binary differential fairness.

    epsilon_ab = max(
        |log P(predicted positive | a) - log P(predicted positive | b)|,
        |log P(predicted negative | a) - log P(predicted negative | b)|,
    )
    """
    p_a = _smoothed_positive_rate(preds_a, threshold, alpha)
    p_b = _smoothed_positive_rate(preds_b, threshold, alpha)

    eps_pos = abs(math.log(p_a) - math.log(p_b))
    eps_neg = abs(math.log(1.0 - p_a) - math.log(1.0 - p_b))

    return max(eps_pos, eps_neg)


def differential_fairness(
    preds: TensorLike,
    group_ids: TensorLike,
    threshold: float = 0.5,
    alpha: float = 1.0,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        aggregate: maximum pairwise epsilon across all group pairs
        epsilon_matrix: [G, G] pairwise DF epsilons
        groups: group IDs in matrix order
        group_rates: [G] smoothed P(predicted positive | g)
    """
    preds_t = _as_1d_tensor(preds, dtype=torch.float32, name="preds")
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")
    _validate_same_length(preds=preds_t, group_ids=group_ids_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")

    if len(groups) == 0:
        empty_vec = torch.empty(0, dtype=torch.float32)
        empty_mat = torch.empty((0, 0), dtype=torch.float32)
        return 0.0, empty_mat, groups, empty_vec

    group_rates = torch.tensor(
        [_smoothed_positive_rate(preds_t[group_ids_t == g], threshold, alpha) for g in groups],
        dtype=torch.float32,
    )

    n_groups = len(groups)
    epsilon_matrix = torch.zeros((n_groups, n_groups), dtype=torch.float32)

    for i, j in itertools.combinations(range(n_groups), 2):
        g1, g2 = groups[i], groups[j]
        preds_g1 = preds_t[group_ids_t == g1]
        preds_g2 = preds_t[group_ids_t == g2]

        eps_ij = differential_fairness_two_groups(
            preds_g1,
            preds_g2,
            threshold=threshold,
            alpha=alpha,
        )

        epsilon_matrix[i, j] = eps_ij
        epsilon_matrix[j, i] = eps_ij

    aggregate = _max_off_diagonal(epsilon_matrix)

    return aggregate, epsilon_matrix, groups, group_rates


# ==================================================
# Probability-vector fairness
# ==================================================

def per_class_fairness_two_groups(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Per-class absolute log-probability gap between two groups.

    Args:
        probs_a: [N_a, C]
        probs_b: [N_b, C]

    Returns:
        epsilon_vector: [C]
    """
    if probs_a.ndim != 2 or probs_b.ndim != 2:
        raise ValueError("probs_a and probs_b must have shape [N, C].")
    if probs_a.shape[1] != probs_b.shape[1]:
        raise ValueError("probs_a and probs_b must have the same number of classes.")

    avg_probs_a = probs_a.float().mean(dim=0)
    avg_probs_b = probs_b.float().mean(dim=0)

    # return torch.abs(torch.log(avg_probs_a.clamp_min(eps)) - torch.log(avg_probs_b.clamp_min(eps)))
    return torch.log(avg_probs_a.clamp_min(eps)) - torch.log(avg_probs_b.clamp_min(eps))


def per_class_fairness(
    probs: TensorLike,
    group_ids: TensorLike,
    *,
    eps: float = 1e-10,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Pairwise per-class log-probability fairness.

    Returns:
        aggregate: max over all group pairs and classes
        tensor: [G, G, C] pairwise class-wise gaps
        groups: group IDs in tensor order
    """
    probs_t = _as_tensor(probs, dtype=torch.float32)
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")

    if probs_t.ndim != 2:
        raise ValueError(f"probs must have shape [N, C], got {tuple(probs_t.shape)}.")
    _validate_same_length(probs=probs_t[:, 0], group_ids=group_ids_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")
    n_groups = len(groups)
    n_classes = probs_t.shape[1]

    tensor = torch.zeros((n_groups, n_groups, n_classes), dtype=torch.float32)

    for i, j in itertools.combinations(range(n_groups), 2):
        g1, g2 = groups[i], groups[j]
        gap = per_class_fairness_two_groups(
            probs_t[group_ids_t == g1],
            probs_t[group_ids_t == g2],
            eps=eps,
        )
        tensor[i, j, :] = gap
        tensor[j, i, :] = gap
    # TODO: Consider if this aggregate makes sense. Because it is not considering the max vector disparity (which is not easy to calculate) but the max element disparity, which may be less meaningful.
    aggregate = float(tensor.max().item()) if tensor.numel() > 0 else 0.0
    return aggregate, tensor, groups


# ==================================================
# Uncertainty-based metrics
# ==================================================

def uncertainty_difference_two_groups(
    uncertainty_a: torch.Tensor,
    uncertainty_b: torch.Tensor,
) -> float:
    """| E[uncertainty | A] - E[uncertainty | B] |."""
    u_a = float(uncertainty_a.float().mean().item()) if uncertainty_a.numel() > 0 else 0.0
    u_b = float(uncertainty_b.float().mean().item()) if uncertainty_b.numel() > 0 else 0.0
    return abs(u_a - u_b)


def uncertainty_difference(
    uncertainties: TensorLike,
    group_ids: TensorLike,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        aggregate: max pairwise uncertainty difference across groups
        matrix: [G, G] pairwise uncertainty differences
        groups: group IDs in matrix order
        group_uncertainties: [G] average uncertainty per group
    """
    uncertainties_t = _as_1d_tensor(uncertainties, dtype=torch.float32, name="uncertainties")
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")
    _validate_same_length(uncertainties=uncertainties_t, group_ids=group_ids_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")

    if len(groups) == 0:
        empty_vec = torch.empty(0, dtype=torch.float32)
        empty_mat = torch.empty((0, 0), dtype=torch.float32)
        return 0.0, empty_mat, groups, empty_vec

    group_uncertainties = torch.tensor(
        [uncertainties_t[group_ids_t == g].mean().item() for g in groups],
        dtype=torch.float32,
    )

    def pairwise_fn(i: int, j: int) -> float:
        g1, g2 = groups[i], groups[j]
        return uncertainty_difference_two_groups(
            uncertainties_t[group_ids_t == g1],
            uncertainties_t[group_ids_t == g2],
        )

    matrix = _build_pairwise_matrix(groups, pairwise_fn, diagonal_value=0.0)
    aggregate = _max_off_diagonal(matrix)

    return aggregate, matrix, groups, group_uncertainties


def subgroup_uncertainty_difference(
    uncertainties: TensorLike,
    group_ids: TensorLike,
    alpha: float = 1.0,
) -> Tuple[float, torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """
    Group-vs-global uncertainty disparity.

    Returns:
        aggregate: max P(g) * |E[U] - E[U|g]|
        values: [G]
        groups: [G]
        global_uncertainty: scalar
        group_uncertainties: [G]
        group_probs: [G]
    """
    uncertainties_t = _as_1d_tensor(uncertainties, dtype=torch.float32, name="uncertainties")
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")
    _validate_same_length(uncertainties=uncertainties_t, group_ids=group_ids_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")

    if len(groups) == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return 0.0, empty, groups, 0.0, empty, empty

    global_uncertainty = float(uncertainties_t.mean().item())

    group_to_idx = {int(g.item()): i for i, g in enumerate(groups)}
    contiguous = torch.tensor(
        [group_to_idx[int(g.item())] for g in group_ids_t],
        dtype=torch.long,
        device=group_ids_t.device,
    )
    _, group_probs = compute_pg_dirichlet_from_groups(contiguous, K=len(groups), alpha=alpha)
    group_probs = group_probs.detach().cpu().float()

    group_uncertainties = torch.tensor(
        [uncertainties_t[group_ids_t == g].mean().item() for g in groups],
        dtype=torch.float32,
    )
    values = group_probs * torch.abs(group_uncertainties - global_uncertainty)
    aggregate = float(values.max().item()) if values.numel() > 0 else 0.0

    return aggregate, values, groups, global_uncertainty, group_uncertainties, group_probs


# ==================================================
# Ensemble-output integration
# ==================================================

def _extract_labels_and_groups_from_loader(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract y and group IDs from a loader yielding `(x, y, g, *rest)`."""
    labels = []
    groups = []

    for batch in loader:
        if len(batch) < 3:
            raise ValueError("Expected loader batches to contain at least `(x, y, group_ids)`.")
        labels.append(_as_tensor(batch[1], dtype=torch.long).view(-1).cpu())
        groups.append(_as_tensor(batch[2], dtype=torch.long).view(-1).cpu())

    if len(labels) == 0:
        raise ValueError("Loader is empty.")

    return torch.cat(labels, dim=0), torch.cat(groups, dim=0)


def evaluate_ensemble_fairness(
    ensemble_outputs: Mapping[str, Any],
    labels: TensorLike,
    group_ids: TensorLike,
    *,
    binary: bool = True,
    threshold: float = 0.5,
    positive_class: int = 1,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate fairness disparities for ensemble predictions and uncertainties.

    Expected `ensemble_outputs` keys from the latest ensemble evaluator:
        - `mean_probs`: [N, C]
        - `predictions`: [N], optional; recomputed from `mean_probs` if absent
        - `predictive_entropy`: [N]
        - `aleatoric_uncertainty`: [N]
        - `epistemic_uncertainty`: [N]

    Returns a nested dictionary with scalar aggregates, matrices/vectors, and group IDs.
    """
    labels_t = _as_1d_tensor(labels, dtype=torch.long, name="labels")
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")

    mean_probs = _as_tensor(ensemble_outputs["mean_probs"], dtype=torch.float32, name="mean_probs")

    positive_scores = positive_class_scores(
        mean_probs,
        positive_class=positive_class,
    )

    if binary:
        # Recompute using the requested threshold, ignoring saved argmax labels.
        is_positive = positive_scores >= threshold
        pred_labels = torch.where(
            is_positive,
            torch.full_like(is_positive, positive_class, dtype=torch.long),
            torch.full_like(is_positive, 1 - positive_class, dtype=torch.long),
        )
    elif "predictions" in ensemble_outputs:
        pred_labels = _as_1d_tensor(
            ensemble_outputs["predictions"],
            dtype=torch.long,
            name="predictions",
        )
    else:
        pred_labels = prediction_scores_to_labels(
            mean_probs,
            positive_class=positive_class,
        )

    sp_agg, sp_matrix, sp_groups = statistical_parity(positive_scores, group_ids_t, threshold=threshold)
    di_agg, di_matrix, di_groups = disparate_impact(positive_scores, group_ids_t, threshold=threshold)
    eo_agg, eo_matrix, eo_groups = equal_opportunity(pred_labels, labels_t, group_ids_t, threshold=threshold, positive_class=positive_class,)

    eodds_agg, eodds_matrix, eodds_groups = equalized_odds(pred_labels, labels_t, group_ids_t, threshold=threshold, positive_class=positive_class,)
    df_agg, df_matrix, df_groups, df_group_rates = differential_fairness(positive_scores, group_ids_t, threshold=threshold, alpha=alpha,)
    ssp = subgroup_statistical_parity(positive_scores, group_ids_t, threshold=threshold,alpha=alpha,)

    per_class = None
    if mean_probs.ndim == 2:
        pc_agg, pc_tensor, pc_groups = per_class_fairness(mean_probs, group_ids_t)
        per_class = {
            "aggregate": pc_agg,
            "tensor": pc_tensor,
            "groups": pc_groups,
        }

    uncertainty_results: Dict[str, Any] = {}
    uncertainty_key_map = {
        "predictive_entropy": "predictive_entropy",
        "aleatoric_uncertainty": "aleatoric_uncertainty",
        "epistemic_uncertainty": "epistemic_uncertainty",
    }

    for input_key, output_key in uncertainty_key_map.items():
        if input_key not in ensemble_outputs:
            continue

        u = _as_1d_tensor(ensemble_outputs[input_key], dtype=torch.float32, name=input_key)
        _validate_same_length(uncertainty=u, group_ids=group_ids_t)

        u_agg, u_matrix, u_groups, u_group_values = uncertainty_difference(u, group_ids_t)
        su = subgroup_uncertainty_difference(u, group_ids_t, alpha=alpha)

        uncertainty_results[output_key] = {
            "pairwise_aggregate": u_agg,
            "pairwise_matrix": u_matrix,
            "groups": u_groups,
            "group_uncertainties": u_group_values,
            "subgroup_aggregate": su[0],
            "subgroup_values": su[1],
            "global_uncertainty": su[3],
            "subgroup_group_uncertainties": su[4],
            "group_probs": su[5],
        }

    return {
        "statistical_parity": {
            "aggregate": sp_agg,
            "matrix": sp_matrix,
            "groups": sp_groups,
        },
        "disparate_impact": {
            "aggregate": di_agg,
            "matrix": di_matrix,
            "groups": di_groups,
        },
        "equal_opportunity": {
            "aggregate": eo_agg,
            "matrix": eo_matrix,
            "groups": eo_groups,
        },
        "equalized_odds": {
            "aggregate": eodds_agg,
            "matrix": eodds_matrix,
            "groups": eodds_groups,
        },
        "differential_fairness": {
            "aggregate": df_agg,
            "matrix": df_matrix,
            "groups": df_groups,
            "group_rates": df_group_rates,
        },
        "subgroup_statistical_parity": {
            "aggregate": ssp[0],
            "values": ssp[1],
            "groups": ssp[2],
            "global_rate": ssp[3],
            "group_rates": ssp[4],
            "group_probs": ssp[5],
        },
        "per_class_fairness": per_class,
        "uncertainty": uncertainty_results,
    }


def evaluate_ensemble_fairness_from_loader(
    ensemble_outputs: Mapping[str, Any],
    loader,
    *,
    binary: bool = True,
    threshold: float = 0.5,
    positive_class: int = 1,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """Convenience wrapper for test loaders yielding `(X, y, group_ids)`."""
    labels, group_ids = _extract_labels_and_groups_from_loader(loader)
    return evaluate_ensemble_fairness(
        ensemble_outputs,
        labels=labels,
        group_ids=group_ids,
        binary=binary,
        threshold=threshold,
        positive_class=positive_class,
        alpha=alpha,
    )


def fairness_summary_to_frame(fairness: Mapping[str, Any]):
    """Convert scalar fairness aggregates to a small pandas DataFrame."""
    if pd is None:
        raise ImportError("pandas is required for fairness_summary_to_frame.")

    rows = []

    for key in [
        "statistical_parity",
        "disparate_impact",
        "equal_opportunity",
        "equalized_odds",
        "differential_fairness",
        "subgroup_statistical_parity",
    ]:
        if key in fairness and fairness[key] is not None:
            rows.append({"metric": key, "value": fairness[key]["aggregate"]})

    for uncertainty_name, item in fairness.get("uncertainty", {}).items():
        rows.append(
            {
                "metric": f"{uncertainty_name}_pairwise_difference",
                "value": item["pairwise_aggregate"],
            }
        )
        rows.append(
            {
                "metric": f"{uncertainty_name}_subgroup_difference",
                "value": item["subgroup_aggregate"],
            }
        )

    if fairness.get("per_class_fairness") is not None:
        rows.append(
            {
                "metric": "per_class_fairness",
                "value": fairness["per_class_fairness"]["aggregate"],
            }
        )

    return pd.DataFrame(rows)


# ==================================================
# Intended usage
# ==================================================

"""
After running the predictive pipeline with an ensemble:

from modules.metrics.fairness_metrics import (
    evaluate_ensemble_fairness_from_loader,
    fairness_summary_to_frame,
)

result = run_predictive_pipeline(config)

fairness = evaluate_ensemble_fairness_from_loader(
    result.ensemble_metrics,
    result.data.test_loader,
    binary=result.data.binary,
    threshold=result.config.train.threshold,
    positive_class=1,
    alpha=1.0,
)

summary = fairness_summary_to_frame(fairness)
print(summary)

# Useful scalar examples:
print(fairness["statistical_parity"]["aggregate"])
print(fairness["uncertainty"]["aleatoric_uncertainty"]["pairwise_aggregate"])
print(fairness["uncertainty"]["epistemic_uncertainty"]["pairwise_aggregate"])

# Save together with experiment outputs:
result.test_metrics["statistical_parity"] = fairness["statistical_parity"]["aggregate"]
"""


# Classical performance-rate comparisons for reports; reuse the existing smoothing.
def performance_fairness(ensemble_outputs, labels, group_ids, *, binary=True,
                         threshold=.5, positive_class=1, alpha=1.):
    """Group confusion counts, rates, gaps and symmetric log-rate ratios.

    rate_epsilon = max_{g,h} |log(rate_g)-log(rate_h)|, with Beta(alpha,alpha)
    smoothing. This is a rate-parity score, distinct from outcome DF, which also
    compares the complementary outcome. Unsupported rates remain None even when
    smoothing is enabled; an aggregate is None if ANY observed group lacks support.
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Performance log-ratios require finite smoothing alpha > 0.')
    probs = _as_tensor(ensemble_outputs['mean_probs'], dtype=torch.float32, name='mean_probs').cpu()
    y = _as_1d_tensor(labels, dtype=torch.long, name='labels').cpu()
    g = _as_1d_tensor(group_ids, dtype=torch.long, name='group_ids').cpu()
    if probs.ndim != 2 or len(probs) != len(y) or len(y) != len(g) or not len(y):
        raise ValueError('Probabilities, labels and groups must be nonempty and aligned.')
    if not 0 <= positive_class < probs.shape[1] or not 0 <= threshold <= 1:
        raise ValueError('Invalid positive class or threshold.')
    if not torch.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError('Invalid probabilities.')
    if binary:
        if probs.shape[1] != 2 or not set(y.tolist()) <= {0,1}:
            raise ValueError('Binary comparisons require two classes and labels 0/1.')
        pred = torch.where(probs[:,positive_class] >= threshold, positive_class, 1-positive_class)
    else:
        pred = probs.argmax(dim=1)
    ids = sorted(g.unique().tolist())
    rows=[]
    events={}
    for gid in ids:
        mask=g==gid
        truth=y[mask]==positive_class; predicted=pred[mask]==positive_class
        tp=int((truth & predicted).sum()); fn=int((truth & ~predicted).sum())
        fp=int((~truth & predicted).sum()); tn=int((~truth & ~predicted).sum())
        selected={
            'TPR':predicted[truth], 'FPR':predicted[~truth],
            'TNR':~predicted[~truth], 'PPV':truth[predicted],
            'Accuracy':pred[mask]==y[mask], 'Selection':predicted,
        }
        row=dict(group_id=int(gid),support=int(mask.sum()),TP=tp,FN=fn,FP=fp,TN=tn)
        events[gid]=selected
        for key,event in selected.items():
            n=event.numel()
            row[key+'_support']=n
            row[key]=float(event.float().mean()) if n else None
            row[key+'_smoothed']=_smoothed_positive_rate(event.float(),.5,alpha) if n else None
        rows.append(row)
    pairs=[]
    metrics=list(events[ids[0]])
    for i,j in itertools.combinations(range(len(ids)),2):
        row=dict(group_a=int(ids[i]),group_b=int(ids[j]))
        for key in metrics:
            a,b=rows[i],rows[j]
            available=a[key] is not None and b[key] is not None
            epsilon=abs(math.log(a[key+'_smoothed'])-math.log(b[key+'_smoothed'])) if available else None
            row[key+'_gap']=abs(a[key]-b[key]) if available else None
            row[key+'_epsilon']=epsilon
            row[key+'_max_ratio']=math.exp(epsilon) if available else None
        pairs.append(row)
    summary={}
    for key in metrics:
        complete=len(ids)>=2 and all(r[key] is not None for r in rows)
        for suffix in ('gap','epsilon','max_ratio'):
            name=key+'_'+suffix
            summary[name]=max(r[name] for r in pairs) if complete else None
    return dict(summary=summary,groups=rows,pairs=pairs,alpha=float(alpha),
                threshold=float(threshold),positive_class=int(positive_class),binary=bool(binary),
                convention='epsilon=max absolute log smoothed rate ratio; max_ratio=exp(epsilon)',
                missing_policy='No eligible observations: null. Aggregate requires support in every observed group.')


def classical_fairness_summary(fairness, performance):
    """Flatten existing outcome-fairness scores and added performance-rate comparisons."""
    names={'statistical_parity':'SP_gap','disparate_impact':'DI_ratio',
           'equal_opportunity':'EO_gap','equalized_odds':'EOdds_gap',
           'differential_fairness':'DF_epsilon','subgroup_statistical_parity':'SSP_gap'}
    result={out:float(fairness[key]['aggregate']) for key,out in names.items()}
    result.update(performance['summary'])
    # Existing EO helpers use zero for unsupported conditional rates; do not report
    # those values as measured fairness when a class is absent from a subgroup.
    if performance['summary']['TPR_gap'] is None:
        result['EO_gap']=None
    if any(performance['summary'][k] is None for k in ('TPR_gap','FPR_gap')):
        result['EOdds_gap']=None
    return result