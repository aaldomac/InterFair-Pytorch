import itertools
import torch
import math

from modules.metrics.performance_metrics import (
    true_positive_rate,
    false_positive_rate,
)
from modules.utils.dataset_utils import (
    compute_pg_dirichlet, 
    compute_pg_dirichlet_from_groups,
)
# ==================================================
# Helpers
# ==================================================

def _positive_prediction_rate(preds: torch.Tensor, threshold: float = 0.5) -> float:
    if preds.numel() == 0:
        return 0.0
    return (preds >= threshold).float().mean().item()


def _group_data(group_ids: torch.Tensor):
    """
    Returns:
        unique_groups: tensor of shape (n_groups,)
    """
    return torch.unique(group_ids)


def _build_pairwise_matrix(
    unique_groups: torch.Tensor,
    pairwise_fn,
    diagonal_value: float,
) -> torch.Tensor:
    """
    Build a symmetric pairwise matrix M where M[i,j] is the pairwise metric
    between group unique_groups[i] and unique_groups[j].

    pairwise_fn(i, j) must return a float.
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
    return matrix[mask].max().item()


def _min_off_diagonal(matrix: torch.Tensor) -> float:
    n = matrix.shape[0]
    if n < 2:
        return 1.0
    mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    return matrix[mask].min().item()


# ==================================================
# Pairwise metrics (2 groups only)
# ==================================================

def statistical_parity_two_groups(
    preds_a: torch.Tensor,
    preds_b: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    | P(\hat{Y}=1 | A) - P(\hat{Y}=1 | B) |
    """
    rate_a = _positive_prediction_rate(preds_a, threshold)
    rate_b = _positive_prediction_rate(preds_b, threshold)
    return abs(rate_a - rate_b)


def equal_opportunity_two_groups(
    preds_a: torch.Tensor,
    labels_a: torch.Tensor,
    preds_b: torch.Tensor,
    labels_b: torch.Tensor,
) -> float:
    """
    | TPR(A) - TPR(B) |
    """
    tpr_a = true_positive_rate(preds_a, labels_a)
    tpr_b = true_positive_rate(preds_b, labels_b)
    return abs(tpr_a - tpr_b)


def disparate_impact_two_groups(
    preds_a: torch.Tensor,
    preds_b: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Symmetric disparate impact ratio:
        min(rate_a, rate_b) / max(rate_a, rate_b)

    Range: [0, 1]
    Best value: 1
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
) -> float:
    """
    max( |TPR(A)-TPR(B)| , |FPR(A)-FPR(B)| )
    """
    tpr_a = true_positive_rate(preds_a, labels_a)
    tpr_b = true_positive_rate(preds_b, labels_b)

    fpr_a = false_positive_rate(preds_a, labels_a)
    fpr_b = false_positive_rate(preds_b, labels_b)

    tpr_diff = abs(tpr_a - tpr_b)
    fpr_diff = abs(fpr_a - fpr_b)

    return max(tpr_diff, fpr_diff)


# ==================================================
# Full metrics returning:
#   aggregate_value, pairwise_matrix, unique_groups
# ==================================================

def statistical_parity(
    preds: torch.Tensor,
    group_ids: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Returns:
        aggregate: max pairwise statistical parity gap
        matrix: [n_groups, n_groups] absolute pairwise gaps
        groups: group labels in matrix order
    """
    groups = _group_data(group_ids)

    if len(groups) < 2:
        matrix = torch.zeros((len(groups), len(groups)), dtype=torch.float32)
        return 0.0, matrix, groups

    def pairwise_fn(i, j):
        g1, g2 = groups[i], groups[j]
        preds_g1 = preds[group_ids == g1]
        preds_g2 = preds[group_ids == g2]
        return statistical_parity_two_groups(preds_g1, preds_g2, threshold)

    matrix = _build_pairwise_matrix(groups, pairwise_fn, diagonal_value=0.0)
    aggregate = _max_off_diagonal(matrix)

    return aggregate, matrix, groups


def equal_opportunity(
    preds: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
):
    """
    Returns:
        aggregate: max pairwise TPR gap
        matrix: [n_groups, n_groups] absolute pairwise TPR gaps
        groups: group labels in matrix order
    """
    groups = _group_data(group_ids)

    if len(groups) < 2:
        matrix = torch.zeros((len(groups), len(groups)), dtype=torch.float32)
        return 0.0, matrix, groups

    def pairwise_fn(i, j):
        g1, g2 = groups[i], groups[j]
        mask1 = (group_ids == g1)
        mask2 = (group_ids == g2)
        return equal_opportunity_two_groups(
            preds[mask1], labels[mask1],
            preds[mask2], labels[mask2],
        )

    matrix = _build_pairwise_matrix(groups, pairwise_fn, diagonal_value=0.0)
    aggregate = _max_off_diagonal(matrix)

    return aggregate, matrix, groups


def disparate_impact(
    preds: torch.Tensor,
    group_ids: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Returns:
        aggregate: worst pairwise DI ratio = minimum off-diagonal entry
        matrix: [n_groups, n_groups] symmetric pairwise DI ratios
        groups: group labels in matrix order

    Notes:
        - diagonal is 1.0
        - lower is worse
        - 1.0 is perfect parity
    """
    groups = _group_data(group_ids)

    if len(groups) < 2:
        matrix = torch.ones((len(groups), len(groups)), dtype=torch.float32)
        return 1.0, matrix, groups

    def pairwise_fn(i, j):
        g1, g2 = groups[i], groups[j]
        preds_g1 = preds[group_ids == g1]
        preds_g2 = preds[group_ids == g2]
        return disparate_impact_two_groups(preds_g1, preds_g2, threshold)

    matrix = _build_pairwise_matrix(groups, pairwise_fn, diagonal_value=1.0)
    aggregate = _min_off_diagonal(matrix)

    return aggregate, matrix, groups


def equalized_odds(
    preds: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
):
    """
    Returns:
        aggregate: max pairwise equalized-odds gap
        matrix: [n_groups, n_groups] pairwise equalized-odds gaps
        groups: group labels in matrix order
    """
    groups = _group_data(group_ids)

    if len(groups) < 2:
        matrix = torch.zeros((len(groups), len(groups)), dtype=torch.float32)
        return 0.0, matrix, groups

    def pairwise_fn(i, j):
        g1, g2 = groups[i], groups[j]
        mask1 = (group_ids == g1)
        mask2 = (group_ids == g2)
        return equalized_odds_two_groups(
            preds[mask1], labels[mask1],
            preds[mask2], labels[mask2],
        )

    matrix = _build_pairwise_matrix(groups, pairwise_fn, diagonal_value=0.0)
    aggregate = _max_off_diagonal(matrix)

    return aggregate, matrix, groups

# INTERSECTIONAL FAIRNESS METRICS #

def subgroup_statistical_parity_fairness_one_group(
    preds_group: torch.Tensor,
    preds_all: torch.Tensor,
    group_probability: float,
    threshold: float = 0.5,
):
    """
    Single-subgroup fairness violation:

        P(g) * | P(\hat{Y}=1) - P(\hat{Y}=1 | g) |

    This is NOT pairwise. It compares one subgroup against the full population.
    """
    global_rate = _positive_prediction_rate(preds_all, threshold)
    group_rate = _positive_prediction_rate(preds_group, threshold)
    return group_probability * abs(global_rate - group_rate)


def subgroup_statistical_parity(
    preds: torch.Tensor,
    group_ids: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Returns:
        aggregate: max subgroup fairness violation
        values: [n_groups] tensor with P(g)*|P(\hat{Y}=1)-P(\hat{Y}=1|g)|
        groups: group labels in the same order as values
        global_rate: P(\hat{Y}=1)
        group_rates: [n_groups] tensor with P(\hat{Y}=1|g)
        group_probs: [n_groups] tensor with P(g)

    Notes:
        - This metric is group-vs-global, not pairwise.
        - Therefore the natural output is a vector, not a matrix.
    """
    groups = torch.unique(group_ids)
    n_total = len(group_ids)

    if len(groups) == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return 0.0, empty, groups, 0.0, empty, empty

    global_rate = _positive_prediction_rate(preds, threshold)

    values = []
    group_rates = []
    group_probs = compute_pg_dirichlet_from_groups(group_ids, groups)

    for g in groups:
        mask = (group_ids == g)
        preds_g = preds[mask]

        values = subgroup_statistical_parity_fairness_one_group(
            preds_g, preds, group_probability=group_probs[len(values)], threshold=threshold
        )
        values.append(values)

    values = torch.tensor(values, dtype=torch.float32)

    aggregate = values.max().item() if len(values) > 0 else 0.0

    return aggregate, values


def _smoothed_positive_rate(
    preds: torch.Tensor,
    threshold: float = 0.5,
    alpha: float = 1.0,
) -> float:
    """
    Smoothed estimate of P(\hat{Y}=1) for binary predictions.

    p = (n_pos + alpha) / (n + 2*alpha)
    """
    n = preds.numel()
    if n == 0:
        return 0.5  # neutral fallback for empty group

    n_pos = (preds >= threshold).float().sum().item()
    return (n_pos + alpha) / (n + 2.0 * alpha)

def differential_fairness_two_groups(
    preds_a: torch.Tensor,
    preds_b: torch.Tensor,
    threshold: float = 0.5,
    alpha: float = 1.0,
) -> float:
    """
    Pairwise epsilon for binary Differential Fairness.

    epsilon_{ab} = max(
        |log P(\hat{Y}=1|a) - log P(\hat{Y}=1|b)|,
        |log P(\hat{Y}=0|a) - log P(\hat{Y}=0|b)|
    )

    Smoothed probabilities are used to avoid log(0).
    """
    p_a = _smoothed_positive_rate(preds_a, threshold, alpha)
    p_b = _smoothed_positive_rate(preds_b, threshold, alpha)

    eps_pos = abs(math.log(p_a) - math.log(p_b))
    eps_neg = abs(math.log(1.0 - p_a) - math.log(1.0 - p_b))

    return max(eps_pos, eps_neg)


def differential_fairness(
    preds: torch.Tensor,
    group_ids: torch.Tensor,
    threshold: float = 0.5,
    alpha: float = 1.0,
):
    """
    Returns:
        aggregate: maximum pairwise epsilon across all group pairs
        epsilon_matrix: [n_groups, n_groups] pairwise DF epsilons
        groups: group labels in matrix order
        group_rates: [n_groups] smoothed P(\hat{Y}=1|g)

    Notes:
        - diagonal is 0
        - larger epsilon means worse fairness
    """
    groups = torch.unique(group_ids)

    if len(groups) == 0:
        empty_vec = torch.empty(0, dtype=torch.float32)
        empty_mat = torch.empty((0, 0), dtype=torch.float32)
        return 0.0, empty_mat, groups, empty_vec

    group_rates = []
    for g in groups:
        preds_g = preds[group_ids == g]
        group_rates.append(_smoothed_positive_rate(preds_g, threshold, alpha))

    group_rates = torch.tensor(group_rates, dtype=torch.float32)

    n_groups = len(groups)
    epsilon_matrix = torch.zeros((n_groups, n_groups), dtype=torch.float32)

    for i, j in itertools.combinations(range(n_groups), 2):
        g1, g2 = groups[i], groups[j]
        preds_g1 = preds[group_ids == g1]
        preds_g2 = preds[group_ids == g2]

        eps_ij = differential_fairness_two_groups(
            preds_g1, preds_g2, threshold=threshold, alpha=alpha
        )

        epsilon_matrix[i, j] = eps_ij
        epsilon_matrix[j, i] = eps_ij

    aggregate = _max_off_diagonal(epsilon_matrix)

    return aggregate, epsilon_matrix, groups, group_rates


# UNCERTAINTY BASED METRICS #

def uncertainty_difference_two_groups(
    uncertainty_a: torch.Tensor,
    uncertainty_b: torch.Tensor,
) -> float:
    """
    Pairwise uncertainty difference based on negative log-probabilities.

    | E[-log P(\hat{Y}|a)] - E[-log P(\hat{Y}|b)] |

    This can be applied to any probabilistic model, including AR models and flows.
    """

    return abs(uncertainty_a - uncertainty_b)

def uncertainty_difference(
    uncertainties: torch.Tensor,
    group_ids: torch.Tensor,
):
    """
    Returns:
        aggregate: max pairwise uncertainty difference across groups
        matrix: [n_groups, n_groups] pairwise uncertainty differences
        groups: group labels in matrix order
        group_uncertainties: [n_groups] average uncertainty per group

    Notes:
        - diagonal is 0
        - larger values mean worse fairness
    """
    groups = torch.unique(group_ids)

    if len(groups) == 0:
        empty_vec = torch.empty(0, dtype=torch.float32)
        empty_mat = torch.empty((0, 0), dtype=torch.float32)
        return 0.0, empty_mat, groups, empty_vec

    group_uncertainties = []
    for g in groups:
        group_uncertainties.append(uncertainties[group_ids == g].mean().item())

    group_uncertainties = torch.tensor(group_uncertainties, dtype=torch.float32)

    n_groups = len(groups)
    matrix = torch.zeros((n_groups, n_groups), dtype=torch.float32)

    for i, j in itertools.combinations(range(n_groups), 2):
        u_i = group_uncertainties[i]
        u_j = group_uncertainties[j]

        diff_ij = uncertainty_difference_two_groups(u_i, u_j)

        matrix[i, j] = diff_ij
        matrix[j, i] = diff_ij

    aggregate = _max_off_diagonal(matrix)

    return aggregate, matrix, groups, group_uncertainties