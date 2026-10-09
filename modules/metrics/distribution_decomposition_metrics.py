"""Compares expectations of predictive distirbutions across groups. Compares envelopes with theoretical epsilon bounds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import itertools

try:
    import pandas as pd
except ImportError:
    pd = None

from modules.metrics.performance_metrics import entropy, kl_divergence, cross_entropy
from modules.utils.tensor_utils import (
        ArrayLike,
        _as_tensor,
        _as_1d_tensor,
        _validate_same_length,
        _unique_sorted_long,
)

# TODO: Consider import TensorLike for substituiting the use of torch.Tensor

# DATA CONTAINERS
@dataclass
class EntropyDecomposition:
    """Entropy decomposition for a set of probability vectors.
    
    For probability vectors p_i and thei mean distribution m=E_i[p_i]:
    H(m) = E_i[H(p_i)] + E_i[KL(p_i||m)]

    Here:
        entropy_of_mean = H(m)
        mean_entropy = E_i[H(p_i)]
        kl_divergence = E_i[KL(p_i||m)]
        residual = H(m) - E_i[H(p_i)] - E_i[KL(p_i||m)]
    """

    mean_probs: torch.Tensor
    entropy_of_mean: float
    mean_entropy: float
    expected_kl_to_mean: float
    residual: float
    n: int

@dataclass
class PairwiseMeanDistributionComparison:
    """Comparison between two group-average predictive distributions."""

    mean_probs_a: torch.Tensor
    mean_probs_b: torch.Tensor
    log_ratio_a_over_b: torch.Tensor
    kl_ab: float
    kl_ba: float
    symmetric_kl: float
    cross_entropy_ab: float
    cross_entropy_ba: float
    entropy_a: float
    entropy_b: float
    signed_expectation_under_a: float
    signed_expectation_under_b: float
    abs_expectation_under_a: float
    abs_expectation_under_b: float


# ==================================================
# Validation helpers
# ==================================================
def _validate_probs_2d(probs: torch.Tensor, *, name: str = "probs") -> torch.Tensor:
    probs = _as_tensor(probs, dtype=torch.float32, name=name, detach=True)

    if probs.ndim != 2:
        raise ValueError(f"{name} must have shape [N, C], got {tuple(probs.shape)}.")
    if probs.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if probs.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two classes, got {probs.shape[1]}.")
    if torch.any(probs < 0):
        raise ValueError(f"{name} contains negative probabilities.")

    return probs


def _normalize_probs_if_needed(
    probs: torch.Tensor,
    *,
    atol: float = 1e-4,
    name: str = "probs",
) -> torch.Tensor:
    """Normalize rows if they are close to, but not exactly, probability distributions."""
    row_sums = probs.sum(dim=-1, keepdim=True)

    if torch.any(row_sums <= 0):
        raise ValueError(f"{name} contains rows with non-positive probability mass.")

    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol, rtol=0.0):
        probs = probs / row_sums

    return probs

# ==================================================
# Core entropy decomposition
# ==================================================
def entropy_decomposition(
        probs: torch.Tensor,
        *,
        eps: float = 1e-10,
        normalize: bool = False,
) -> EntropyDecomposition:
    """Compute the decomposition H(E[p])=E[H(p)]+E[KL(p||E[9])]
    Args:
        probs: Probability vectors with shape [N, C]
        eps: Numerical epsilon.
        normalize: If True, normalize rows before computing metrics.

    Returns:
        EntropyDecomposition.
    """
    probs_t = _validate_probs_2d(probs)
    if normalize:
        probs_t = _normalize_probs_if_needed(probs_t)

    mean_probs = probs_t.mean(dim=0)

    entropy_of_mean_t = entropy(mean_probs.unsqueeze(0), epsilon=eps).squeeze(0)
    per_sample_entropy = entropy(probs_t, epsilon=eps)
    mean_entropy_t = per_sample_entropy.mean()

    mean_repeated = mean_probs.unsqueeze(0).expand_as(probs_t)
    expected_kl_t = kl_divergence(probs_t, mean_repeated, epsilon=eps).mean()

    residual_t = entropy_of_mean_t - mean_entropy_t - expected_kl_t  # should be zero

    return EntropyDecomposition(
        mean_probs=mean_probs,
        entropy_of_mean=float(entropy_of_mean_t.item()),
        mean_entropy=float(mean_entropy_t.item()),
        expected_kl_to_mean=float(expected_kl_t.item()),
        residual=float(residual_t.item()),
        n=int(probs_t.shape[0]),
    )

def assert_entropy_decomposition(
    probs: torch.Tensor,
    *,
    atol: float = 1e-6,
    eps: float = 1e-10,
    normalize: bool = False,
) -> EntropyDecomposition:
    """Compute and assert the entropy decomposition up to a tolerance."""
    result = entropy_decomposition(probs, eps=eps, normalize=normalize)

    if abs(result.residual) > atol:
        raise AssertionError(
            "Entropy decomposition failed: "
            f"H(mean)={result.entropy_of_mean:.8f}, "
            f"mean H={result.mean_entropy:.8f}, "
            f"mean KL={result.expected_kl_to_mean:.8f}, "
            f"residual={result.residual:.8e}, atol={atol:.1e}."
        )

    return result

# ==================================================
# Pairwise group-average distribution comparison
# ==================================================
def compare_mean_distributions(
        probs_a: torch.Tensor,
        probs_b: torch.Tensor,
        *,
        eps: float = 1e-10,
        normalize: bool = False,
) -> PairwiseMeanDistributionComparison:
    """Compare two groups through their average predictive distirbution.
    Let:
        m_A = E[p(x) | A]
        m_B = E[p(x) | B]
        r = log(m_A / m_B)

    Then:
        E_{c ~ m_A}[r_c] = KL(m_A || m_B)
        E_{c ~ m_B}[r_c] = -KL(m_B || m_A)

    Your previous `average_epsilon_A` is `abs(E_{m_A}[r])`, which equals KL(A || B).
    Your previous `average_epsilon_B` is `abs(E_{m_B}[r])`, which equals KL(B || A).
    """
    probs_a_t = _validate_probs_2d(probs_a, name="probs_a")
    probs_b_t = _validate_probs_2d(probs_b, name="probs_b")

    if probs_a_t.shape[1] != probs_b_t.shape[1]:
        raise ValueError(
            f"Both groups must have the same number of classes. "
            f"Got {probs_a_t.shape[1]} and {probs_b_t.shape[1]}."
        )

    if normalize:
        probs_a_t = _normalize_probs_if_needed(probs_a_t, name="probs_a")
        probs_b_t = _normalize_probs_if_needed(probs_b_t, name="probs_b")

    mean_a = probs_a_t.mean(dim=0)
    mean_b = probs_b_t.mean(dim=0)

    mean_a_safe = mean_a.clamp_min(eps)
    mean_b_safe = mean_b.clamp_min(eps)

    log_ratio = torch.log(mean_a_safe / mean_b_safe)  # analogous of the fairness epsilon

    kl_ab_t = kl_divergence(mean_a.unsqueeze(0), mean_b.unsqueeze(0), epsilon=eps).squeeze(0)
    kl_ba_t = kl_divergence(mean_b.unsqueeze(0), mean_a.unsqueeze(0), epsilon=eps).squeeze(0)

    ce_ab_t = cross_entropy(mean_a.unsqueeze(0), mean_b.unsqueeze(0), epsilon=eps).squeeze(0)
    ce_ba_t = cross_entropy(mean_b.unsqueeze(0), mean_a.unsqueeze(0), epsilon=eps).squeeze(0)

    h_a_t = entropy(mean_a.unsqueeze(0), epsilon=eps).squeeze(0)
    h_b_t = entropy(mean_b.unsqueeze(0), epsilon=eps).squeeze(0)

    signed_a_t = torch.sum(mean_a * log_ratio)  # KL = E_{y|g}[epsilon]
    signed_b_t = torch.sum(mean_b * log_ratio)  # -KL = -E_{y|g'}[epsilon]

    return PairwiseMeanDistributionComparison(
        mean_probs_a=mean_a,
        mean_probs_b=mean_b,
        log_ratio_a_over_b=log_ratio,
        kl_ab=float(kl_ab_t.item()),
        kl_ba=float(kl_ba_t.item()),
        symmetric_kl=float((kl_ab_t + kl_ba_t).item()),
        cross_entropy_ab=float(ce_ab_t.item()),
        cross_entropy_ba=float(ce_ba_t.item()),
        entropy_a=float(h_a_t.item()),
        entropy_b=float(h_b_t.item()),
        signed_expectation_under_a=float(signed_a_t.item()),
        signed_expectation_under_b=float(signed_b_t.item()),
        abs_expectation_under_a=float(abs(signed_a_t.item())),
        abs_expectation_under_b=float(abs(signed_b_t.item())),
    )

def mean_distribution_kl(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    *,
    eps: float = 1e-10,
) -> Tuple[float, float]:
    """Return KL(mean_A || mean_B) and KL(mean_B || mean_A)."""
    comparison = compare_mean_distributions(probs_a, probs_b, eps=eps)
    return comparison.kl_ab, comparison.kl_ba

# ==================================================
# Group-level summaries
# ==================================================
def group_entropy_decompositions(
        probs: torch.Tensor,
        group_ids: torch.Tensor,
        *,
        eps: float = 1e-10,
        normalize: bool = False,
) -> Dict[int, EntropyDecomposition]:
    """Compute entropy decomposition separately for every group."""
    probs_t = _validate_probs_2d(probs)
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")
    _validate_same_length(probs=probs_t[:, 0], group_ids=group_ids_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")

    results: Dict[int, EntropyDecomposition] = {}
    for group in groups:
        gid = int(group.item())
        mask = group_ids_t == group
        results[gid] = entropy_decomposition(
            probs_t[mask],
            eps=eps,
            normalize=normalize,
        )
    
    return results

def group_entropy_decompositions_to_frame(
    decompositions: Mapping[int, EntropyDecomposition],
):
    """Convert group entropy decompositions to a pandas DataFrame."""
    if pd is None:
        raise ImportError("pandas is required for group_entropy_decompositions_to_frame.")

    rows = []
    for group_id, dec in decompositions.items():
        rows.append(
            {
                "group_id": group_id,
                "n": dec.n,
                "entropy_of_mean": dec.entropy_of_mean,
                "mean_entropy": dec.mean_entropy,
                "expected_kl_to_mean": dec.expected_kl_to_mean,
                "residual": dec.residual,
            }
        )

    return pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)

def pairwise_group_mean_distribution_comparisons(
        probs: torch.Tensor,
        group_ids: torch.Tensor,
        *,
        eps: float = 1e-10,
        normalize: bool = False,
) -> Dict[Tuple[int, int], PairwiseMeanDistributionComparison]:
    """Compare every pair of groups through their average predictive distirbutions."""
    probs_t = _validate_probs_2d(probs)
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")
    _validate_same_length(probs=probs_t[:, 0], group_ids=group_ids_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")

    results: Dict[Tuple[int, int], PairwiseMeanDistributionComparison] = {}

    for g_i, g_j in itertools.combinations(groups.tolist(), 2):
        g_i_int = int(g_i)
        g_j_int = int(g_j)
        probs_i = probs_t[group_ids_t == g_i_int]
        probs_j = probs_t[group_ids_t == g_j_int]

        results[(g_i_int, g_j_int)] = compare_mean_distributions(
            probs_i,
            probs_j,
            eps=eps,
            normalize=normalize,
        )

    return results

def pairwise_group_mean_distribution_matrices(
    probs: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    eps: float = 1e-10,
    normalize: bool = False,
) -> Dict[str, torch.Tensor]:
    """Return pairwise matrices for group-average predictive distributions.

    Returns:
        groups: [G]
        kl: [G, G], where kl[i, j] = KL(mean_i || mean_j)
        symmetric_kl: [G, G], where symmetric_kl[i, j] = KL(i||j) + KL(j||i)
        cross_entropy: [G, G], where cross_entropy[i, j] = H(mean_i, mean_j)
        entropy_of_mean: [G], where entropy_of_mean[i] = H(mean_i)
        class_log_ratio: [G, G, C], where class_log_ratio[i, j, c] = log(mean_i[c] / mean_j[c])
    """
    probs_t = _validate_probs_2d(probs)
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")
    _validate_same_length(probs=probs_t[:, 0], group_ids=group_ids_t)

    if normalize:
        probs_t = _normalize_probs_if_needed(probs_t)

    groups = _unique_sorted_long(group_ids_t, name="group_ids")
    n_groups = len(groups)
    n_classes = probs_t.shape[1]

    group_means = []
    for group in groups:
        group_means.append(probs_t[group_ids_t == group].mean(dim=0))
    group_means_t = torch.stack(group_means, dim=0)  # [G, C]

    kl_matrix = torch.zeros((n_groups, n_groups), dtype=torch.float32)
    ce_matrix = torch.zeros((n_groups, n_groups), dtype=torch.float32)
    class_log_ratio = torch.zeros((n_groups, n_groups, n_classes), dtype=torch.float32)

    for i, j in itertools.product(range(n_groups), range(n_groups)):
        p_i = group_means_t[i].unsqueeze(0)
        p_j = group_means_t[j].unsqueeze(0)

        kl_matrix[i, j] = kl_divergence(p_i, p_j, epsilon=eps).squeeze(0)
        ce_matrix[i, j] = cross_entropy(p_i, p_j, epsilon=eps).squeeze(0)
        class_log_ratio[i, j, :] = torch.log(group_means_t[i].clamp_min(eps) / group_means_t[j].clamp_min(eps))

    entropy_of_mean = entropy(group_means_t, epsilon=eps)
    symmetric_kl = kl_matrix + kl_matrix.T

    return {
        "groups": groups,
        "group_mean_probs": group_means_t,
        "kl": kl_matrix,
        "symmetric_kl": symmetric_kl,
        "cross_entropy": ce_matrix,
        "entropy_of_mean": entropy_of_mean,
        "class_log_ratio": class_log_ratio,
    }

# ==================================================
# Link with latest ensemble outputs
# ==================================================
def group_distribution_analysis_from_ensemble_outputs(
    ensemble_outputs: Mapping[str, Any],
    group_ids: torch.Tensor,
    *,
    probs_key: str = "mean_probs",
    eps: float = 1e-10,
) -> Dict[str, Any]:
    """Run group distribution analysis using ensemble mean probabilities.
    
    This is meant to consume the output of 'evaluate_ensemble(...)':
        {
            "mean_probs": [N, C],
            "predictive_entropy": [N],
            "aleatoric_entropy": [N],
            "epistemic_uncertainty": [N],
            ...
        }
    """
    if probs_key not in ensemble_outputs:
        raise KeyError(f"ensemble_outputs must contain key '{probs_key}'.")
    
    probs = _as_tensor(ensemble_outputs[probs_key], dtype=torch.float32, name=probs_key)
    group_ids_t = _as_1d_tensor(group_ids, dtype=torch.long, name="group_ids")

    decompositions = group_entropy_decompositions(probs, group_ids_t, eps=eps)
    matrices = pairwise_group_mean_distribution_matrices(probs, group_ids_t, eps=eps)

    return {
        "group_entropy_decompositions": decompositions,
        "pairwise_group_mean_distribution_matrices": matrices,
    }

"""
Example usage with the latest pipeline result:

from modules.metrics.distribution_decomposition_metrics import (
    group_distribution_analysis_from_ensemble_outputs,
    group_entropy_decompositions_to_frame,
)

labels, group_ids = [], []
for x, y, g in result.data.test_loader:
    labels.append(y)
    group_ids.append(g)
group_ids = torch.cat(group_ids)

analysis = group_distribution_analysis_from_ensemble_outputs(
    result.ensemble_metrics,
    group_ids=group_ids,
)

decomp_df = group_entropy_decompositions_to_frame(
    analysis["group_entropy_decompositions"]
)
print(decomp_df)

matrices = analysis["pairwise_group_mean_distribution_matrices"]
print(matrices["kl"])
print(matrices["symmetric_kl"])
print(matrices["class_log_ratio"].shape)  # [num_groups, num_groups, num_classes]


Example usage for two manually selected groups:

from modules.metrics.distribution_decomposition_metrics import (
    assert_entropy_decomposition,
    compare_mean_distributions,
)

A = mean_probs[group_ids == group_A]
B = mean_probs[group_ids == group_B]

A_decomp = assert_entropy_decomposition(A)
B_decomp = assert_entropy_decomposition(B)

print(A_decomp.entropy_of_mean)
print(A_decomp.mean_entropy)
print(A_decomp.expected_kl_to_mean)

comparison = compare_mean_distributions(A, B)

print(comparison.log_ratio_a_over_b)
print(comparison.kl_ab)
print(comparison.kl_ba)
print(comparison.abs_expectation_under_a)  # equals KL(A || B)
print(comparison.abs_expectation_under_b)  # equals KL(B || A)
"""
