from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

try:
    from scipy.stats import spearmanr, kendalltau, pearsonr, rankdata
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float], Sequence[int]]


# ==================================================
# Generic conversion and validation helpers
# ==================================================

def _to_numpy(x: Any, *, dtype: Optional[type] = float) -> np.ndarray:
    """Convert tensors/sequences to NumPy arrays."""
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if dtype is not None:
        arr = arr.astype(dtype)

    return arr


def _to_group_names(group_names: Any) -> list[str]:
    if torch.is_tensor(group_names):
        group_names = group_names.detach().cpu().numpy()
    return [str(g) for g in list(group_names)]


def _validate_square_matrix(matrix: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {matrix.shape}.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}.")

    return matrix


def _validate_same_shape(matrix_a: np.ndarray, matrix_b: np.ndarray) -> None:
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(
            f"Both matrices must have the same shape. Got {matrix_a.shape} and {matrix_b.shape}."
        )


def _triangle_indices(matrix: np.ndarray, triangle: str = "upper") -> tuple[np.ndarray, np.ndarray]:
    if triangle == "upper":
        return np.triu_indices_from(matrix, k=1)
    if triangle == "lower":
        return np.tril_indices_from(matrix, k=-1)
    if triangle == "both":
        n = matrix.shape[0]
        return np.where(~np.eye(n, dtype=bool))
    raise ValueError("triangle must be 'upper', 'lower', or 'both'.")

def extract_pairwise_values(
        matrix: ArrayLike,
        *,
        triangle: str = "upper",
        drop_nan: bool = True,
) -> np.ndarray:
    """Extract off-diagonal pairwise values from a square matrix."""
    matrix_np = _validate_square_matrix(_to_numpy(matrix), name="matrix")
    idx = _triangle_indices(matrix_np, triangle=triangle)
    values = matrix_np[idx].astype(float)

    if drop_nan:
        values = values[np.isfinite(values)]

    return values

def extract_pairwise_dataframe(
        matrix: ArrayLike,
        group_names: Sequence[Any],
        *,
        metric_name: str,
        triangle: str = "upper",
) -> pd.DataFrame:
    """Return pairwise matrix entries as a DataFrame with group names.
    Useful for debugging, saving, or joining classical and uncertainty metrics.
    """
    if pd is None:
        raise ImportError("pandas is required for extract_pairwise_dataframe.")

    matrix_np = _validate_square_matrix(_to_numpy(matrix), name="matrix")
    names = _to_group_names(group_names)

    if len(names) != matrix_np.shape[0]:
        raise ValueError(
            f"Expected {matrix_np.shape[0]} group names, got {len(names)}."
        )

    idx_i, idx_j = _triangle_indices(matrix_np, triangle=triangle)

    rows = []
    for i, j in zip(idx_i, idx_j):
        value = matrix_np[i, j]
        if not np.isfinite(value):
            continue
        rows.append(
            {
                "group_i": names[i],
                "group_j": names[j],
                "i": int(i),
                "j": int(j),
                metric_name: float(value),
            }
        )

    return pd.DataFrame(rows)

def _safe_minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    out = np.full_like(values, np.nan, dtype=float)
    if not finite.any():
        return out

    v_min = np.nanmin(values[finite])
    v_max = np.nanmax(values[finite])

    if np.isclose(v_min, v_max):
        out[finite] = 0.0
    else:
        out[finite] = (values[finite] - v_min) / (v_max - v_min)

    return out

def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if _SCIPY_AVAILABLE:
        return rankdata(values, method="average") - 1.0

    if pd is not None:
        return pd.Series(values).rank(method="average").to_numpy() - 1.0

    # Fallback: ordinal ranks. Ties are not averaged.
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks

def matrix_correlation(
    matrix_a: ArrayLike,
    matrix_b: ArrayLike,
    *,
    correlation: str = "spearman",
    triangle: str = "upper",
) -> Dict[str, Any]:
    """Compute correlation between corresponding pairwise entries of two matrices."""
    a = _validate_square_matrix(_to_numpy(matrix_a), name="matrix_a")
    b = _validate_square_matrix(_to_numpy(matrix_b), name="matrix_b")
    _validate_same_shape(a, b)

    idx = _triangle_indices(a, triangle=triangle)
    values_a = a[idx].astype(float)
    values_b = b[idx].astype(float)

    valid = np.isfinite(values_a) & np.isfinite(values_b)
    values_a = values_a[valid]
    values_b = values_b[valid]

    if len(values_a) < 2:
        raise ValueError("At least two valid pairwise entries are required to compute correlation.")

    correlation = correlation.lower()
    if correlation not in {"spearman", "kendall", "pearson"}:
        raise ValueError("correlation must be 'spearman', 'kendall', or 'pearson'.")

    if _SCIPY_AVAILABLE:
        if correlation == "spearman":
            corr_value, p_value = spearmanr(values_a, values_b)
        elif correlation == "kendall":
            corr_value, p_value = kendalltau(values_a, values_b)
        else:
            corr_value, p_value = pearsonr(values_a, values_b)
    else:
        if correlation == "kendall":
            raise ImportError("scipy is required for Kendall correlation.")
        if correlation == "spearman":
            corr_value = np.corrcoef(_rank(values_a), _rank(values_b))[0, 1]
        else:
            corr_value = np.corrcoef(values_a, values_b)[0, 1]
        p_value = np.nan

    return {
        "correlation_type": correlation,
        "correlation_value": float(corr_value),
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "values_a": values_a,
        "values_b": values_b,
        "rank_a": _rank(values_a),
        "rank_b": _rank(values_b),
        "triangle": triangle,
        "n_pairs": int(len(values_a)),
    }


# ==================================================
# Matrix heatmaps
# ==================================================
def plot_fairness_matrix(
    fairness_matrix: ArrayLike,
    group_names: Sequence[Any],
    metric_name: str,
    *,
    annotate: bool = False,
    annotation_fmt: str = ".2f",
    show_x_labels: bool = True,
    show_y_labels: bool = True,
    mask_diagonal: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple[float, float] = (8, 6),
):
    """
    Plot a square group-pair fairness/disparity matrix as a heatmap
    """
    matrix = _validate_square_matrix(_to_numpy(fairness_matrix), name="fairness_matrix")
    names = _to_group_names(group_names)

    if len(names) != matrix.shape[0]:
        raise ValueError(f"Expected {matrix.shape[0]} group names, got {len(names)}.")

    matrix_plot = matrix.copy()
    if mask_diagonal:
        np.fill_diagonal(matrix_plot, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix_plot, aspect="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)

    n = matrix.shape[0]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    if show_x_labels:
        ax.set_xticklabels(names, rotation=45, ha="right")
    else:
        ax.set_xticklabels([])

    if show_y_labels:
        ax.set_yticklabels(names)
    else:
        ax.set_yticklabels([])

    ax.set_xlabel("Group")
    ax.set_ylabel("Group")
    ax.set_title(f"{metric_name} matrix")

    if annotate:
        for i in range(n):
            for j in range(n):
                value = matrix_plot[i, j]
                if np.isfinite(value):
                    ax.text(j, i, format(value, annotation_fmt), ha="center", va="center")

    fig.tight_layout()
    return fig



def mix_fairness_matrices(
    matrix_a: ArrayLike,
    matrix_b: ArrayLike,
    *,
    diagonal_value: float = np.nan,
    normalize: Optional[str] = "separate",
) -> np.ndarray:
    """
    Mix two square matrices.

    Upper triangle comes from matrix_a. Lower triangle comes from matrix_b.

    Args:
        matrix_a: First square matrix.
        matrix_b: Second square matrix.
        diagonal_value: Value placed on the diagonal.
        normalize:
            - None: no normalization
            - "separate": normalize upper and lower triangles independently to [0, 1]
            - "global": normalize all off-diagonal values jointly to [0, 1]

    Returns:
        Mixed matrix.
    """
    a = _validate_square_matrix(_to_numpy(matrix_a), name="matrix_a")
    b = _validate_square_matrix(_to_numpy(matrix_b), name="matrix_b")
    _validate_same_shape(a, b)

    mixed = np.full_like(a, np.nan, dtype=float)
    upper = np.triu_indices_from(mixed, k=1)
    lower = np.tril_indices_from(mixed, k=-1)

    upper_values = a[upper].astype(float)
    lower_values = b[lower].astype(float)

    if normalize is None:
        mixed[upper] = upper_values
        mixed[lower] = lower_values
    elif normalize == "separate":
        mixed[upper] = _safe_minmax_normalize(upper_values)
        mixed[lower] = _safe_minmax_normalize(lower_values)
    elif normalize == "global":
        both = np.concatenate([upper_values, lower_values])
        both_norm = _safe_minmax_normalize(both)
        mixed[upper] = both_norm[: len(upper_values)]
        mixed[lower] = both_norm[len(upper_values) :]
    else:
        raise ValueError("normalize must be None, 'separate', or 'global'.")

    np.fill_diagonal(mixed, diagonal_value)
    return mixed

def plot_mixed_fairness_matrix(
    matrix_a: ArrayLike,
    matrix_b: ArrayLike,
    group_names: Sequence[Any],
    *,
    label_a: str = "Metric A",
    label_b: str = "Metric B",
    normalize: Optional[str] = "separate",
    annotate: bool = False,
    figsize: tuple[float, float] = (8, 6),
):
    """Plot upper-triangle/lower-triangle comparison of two matrices."""
    mixed = mix_fairness_matrices(matrix_a, matrix_b, normalize=normalize)
    norm_text = "" if normalize is None else f" ({normalize} normalized)"

    fig = plot_fairness_matrix(
        mixed,
        group_names,
        metric_name=f"Upper: {label_a}; lower: {label_b}{norm_text}",
        annotate=annotate,
        mask_diagonal=False,
        figsize=figsize,
    )
    return fig
 
# ==================================================
# Matrix comparison scatter/order plots
# ==================================================
def scatter_fairness_matrices(
    matrix_a,
    matrix_b,
    correlation="spearman",
    plot_mode="ranks",
    triangle="upper",
    normalize=False,
    labels=("Metric A", "Metric B"),
    return_data=False,
):
    """
    Compare two square fairness matrices through corresponding pairwise entries.

    Useful for checking whether classical fairness disparities and uncertainty
    disparities rank group pairs similarly.

    plot_mode:
        - "ranks": rank_a vs rank_b
        - "values": raw or normalized metric_a vs metric_b
        - "sorted_values": sort by metric A and plot both sequences
        - "order_positions": compare sorted permutations
    """
    stats = matrix_correlation(
        matrix_a,
        matrix_b,
        correlation=correlation,
        triangle=triangle,
    )

    values_a = stats["values_a"]
    values_b = stats["values_b"]
    rank_a = stats["rank_a"]
    rank_b = stats["rank_b"]
    order_a = np.argsort(values_a)
    order_b = np.argsort(values_b)

    if normalize:
        values_a_plot = _safe_minmax_normalize(values_a)
        values_b_plot = _safe_minmax_normalize(values_b)
    else:
        values_a_plot = values_a.copy()
        values_b_plot = values_b.copy()

    corr_value = stats["correlation_value"]

    fig, ax = plt.subplots(figsize=(8, 6))

    if plot_mode == "ranks":
        ax.scatter(rank_a, rank_b, alpha=0.7)
        lim = max(rank_a.max(), rank_b.max()) if len(rank_a) else 1
        ax.plot([0, lim], [0, lim], linestyle="--", alpha=0.6)
        ax.set_xlabel(f"Rank in {labels[0]}")
        ax.set_ylabel(f"Rank in {labels[1]}")
        ax.set_title(f"Pairwise ordering comparison ({correlation} = {corr_value:.4f})")

    elif plot_mode == "values":
        ax.scatter(values_a_plot, values_b_plot, alpha=0.7)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(f"Pairwise value comparison ({correlation} = {corr_value:.4f})")

    elif plot_mode == "sorted_values":
        sorted_a = values_a_plot[order_a]
        sorted_b_by_a = values_b_plot[order_a]
        x = np.arange(len(sorted_a))
        ax.scatter(x, sorted_a, alpha=0.7, label=labels[0])
        ax.scatter(x, sorted_b_by_a, alpha=0.7, label=labels[1])
        ax.set_xlabel(f"Pair index sorted by {labels[0]}")
        ax.set_ylabel("Normalized value" if normalize else "Value")
        ax.set_title(f"Sorted pairwise values ({correlation} = {corr_value:.4f})")
        ax.legend()

    elif plot_mode == "order_positions":
        x = np.arange(len(order_a))
        ax.scatter(x, order_a, alpha=0.7, label=labels[0])
        ax.scatter(x, order_b, alpha=0.7, label=labels[1])
        ax.set_xlabel("Sorted position")
        ax.set_ylabel("Original pair index")
        ax.set_title(f"Permutation comparison ({correlation} = {corr_value:.4f})")
        ax.legend()

    else:
        raise ValueError("plot_mode must be 'ranks', 'values', 'sorted_values', or 'order_positions'.")

    ax.grid(alpha=0.3)
    fig.tight_layout()

    if return_data:
        return fig, stats

def plot_pairwise_metric_correlation_matrix(
    matrices: Mapping[str, ArrayLike],
    *,
    correlation: str = "spearman",
    triangle: str = "upper",
    annotate: bool = True,
    figsize: tuple[float, float] = (8, 6),
):
    """Plot correlation between many pairwise fairness/uncertainty matrices.

    Example:
        matrices = {
            "statistical_parity": fairness["statistical_parity"]["matrix"],
            "equalized_odds": fairness["equalized_odds"]["matrix"],
            "epistemic": fairness["uncertainty"]["epistemic_uncertainty"]["pairwise_matrix"],
            "aleatoric": fairness["uncertainty"]["aleatoric_entropy"]["pairwise_matrix"],
        }
    """
    names = list(matrices.keys())
    if len(names) < 2:
        raise ValueError("At least two matrices are required.")

    corr = np.eye(len(names), dtype=float)

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i >= j:
                continue
            stats = matrix_correlation(
                matrices[name_i],
                matrices[name_j],
                correlation=correlation,
                triangle=triangle,
            )
            corr[i, j] = stats["correlation_value"]
            corr[j, i] = stats["correlation_value"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, aspect="auto", vmin=-1.0, vmax=1.0)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title(f"Pairwise metric correlation matrix ({correlation})")

    if annotate:
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")

    fig.tight_layout()
    return fig, {"matrix": corr, "metric_names": names, "correlation": correlation}

# ==================================================
# Group-level plots
# ==================================================
def plot_group_values(
    group_names: Sequence[Any],
    values: ArrayLike,
    *,
    title: str,
    ylabel: str,
    sort: bool = True,
    figsize: tuple[float, float] = (9, 5),
):
    """Bar plot of one value per group."""
    names = _to_group_names(group_names)
    values_np = _to_numpy(values).reshape(-1)

    if len(names) != len(values_np):
        raise ValueError(f"Expected {len(values_np)} group names, got {len(names)}.")

    order = np.argsort(values_np) if sort else np.arange(len(values_np))
    values_plot = values_np[order]
    names_plot = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(len(values_plot)), values_plot)
    ax.set_xticks(np.arange(len(values_plot)))
    ax.set_xticklabels(names_plot, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig

def plot_group_metric_scatter(
    x_values: ArrayLike,
    y_values: ArrayLike,
    group_names: Sequence[Any],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    annotate: bool = True,
    correlation: str = "spearman",
    figsize: tuple[float, float] = (7, 5),
):
    """Scatter one group-level quantity against another.

    Useful examples:
        - group accuracy vs epistemic uncertainty
        - group positive rate vs aleatoric uncertainty
        - group TPR vs predictive entropy
    """
    x = _to_numpy(x_values).reshape(-1)
    y = _to_numpy(y_values).reshape(-1)
    names = _to_group_names(group_names)

    if not (len(x) == len(y) == len(names)):
        raise ValueError(f"Mismatched lengths: x={len(x)}, y={len(y)}, groups={len(names)}.")

    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]
    names_valid = [name for name, keep in zip(names, valid) if keep]

    if len(x_valid) >= 2:
        if _SCIPY_AVAILABLE:
            if correlation == "spearman":
                corr_value, p_value = spearmanr(x_valid, y_valid)
            elif correlation == "kendall":
                corr_value, p_value = kendalltau(x_valid, y_valid)
            elif correlation == "pearson":
                corr_value, p_value = pearsonr(x_valid, y_valid)
            else:
                raise ValueError("correlation must be 'spearman', 'kendall', or 'pearson'.")
        else:
            if correlation == "kendall":
                raise ImportError("scipy is required for Kendall correlation.")
            corr_value = np.corrcoef(_rank(x_valid) if correlation == "spearman" else x_valid, _rank(y_valid) if correlation == "spearman" else y_valid)[0, 1]
            p_value = np.nan
    else:
        corr_value, p_value = np.nan, np.nan

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_valid, y_valid, alpha=0.8)

    if annotate:
        for xi, yi, name in zip(x_valid, y_valid, names_valid):
            ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({correlation} = {corr_value:.4f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig, {
        "correlation_type": correlation,
        "correlation_value": float(corr_value) if np.isfinite(corr_value) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "x_values": x_valid,
        "y_values": y_valid,
        "group_names": names_valid,
    }

# ==================================================
# Metric summary plots
# ==================================================

def plot_metric_summary(
    metrics: Mapping[str, float],
    *,
    title: str = "Metric summary",
    ylabel: str = "Value",
    sort: bool = False,
    figsize: tuple[float, float] = (9, 5),
):
    """Bar plot for scalar metrics."""
    names = list(metrics.keys())
    values = np.asarray([float(metrics[name]) for name in names], dtype=float)

    order = np.argsort(values) if sort else np.arange(len(values))
    names_plot = [names[i] for i in order]
    values_plot = values[order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(len(values_plot)), values_plot)
    ax.set_xticks(np.arange(len(values_plot)))
    ax.set_xticklabels(names_plot, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_fairness_summary_from_dict(
    fairness: Mapping[str, Any],
    *,
    include_uncertainty: bool = True,
    title: str = "Fairness and uncertainty disparity summary",
    figsize: tuple[float, float] = (10, 5),
):
    """Create a scalar summary plot from the fairness dictionary returned by `evaluate_ensemble_fairness`.

    Notes:
        Disparate impact has the opposite direction from gap metrics: lower is worse,
        while most other values are higher-is-worse. Interpret it separately.
    """
    metrics: Dict[str, float] = {}

    for key in [
        "statistical_parity",
        "equal_opportunity",
        "equalized_odds",
        "differential_fairness",
        "subgroup_statistical_parity",
    ]:
        if key in fairness and fairness[key] is not None:
            metrics[key] = float(fairness[key]["aggregate"])

    if "disparate_impact" in fairness and fairness["disparate_impact"] is not None:
        metrics["disparate_impact"] = float(fairness["disparate_impact"]["aggregate"])

    if include_uncertainty:
        for uncertainty_name, item in fairness.get("uncertainty", {}).items():
            metrics[f"{uncertainty_name}_pairwise"] = float(item["pairwise_aggregate"])
            metrics[f"{uncertainty_name}_subgroup"] = float(item["subgroup_aggregate"])

    return plot_metric_summary(metrics, title=title, ylabel="Aggregate value", sort=False, figsize=figsize)

# ==================================================
# Distribution decomposition plots
# ==================================================

def plot_entropy_decomposition_by_group(
    decomposition_table: Any,
    *,
    group_col: str = "group_id",
    title: str = "Group entropy decomposition",
    figsize: tuple[float, float] = (10, 5),
):
    """Plot H(mean), mean H(p), and mean KL(p || mean) by group.

    Expected columns:
        group_id, entropy_of_mean, mean_entropy, expected_kl_to_mean
    """
    if pd is None:
        raise ImportError("pandas is required for plot_entropy_decomposition_by_group.")

    df = decomposition_table.copy()
    required = [group_col, "entropy_of_mean", "mean_entropy", "expected_kl_to_mean"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, df["entropy_of_mean"].to_numpy(), width=width, label="H(group mean)")
    ax.bar(x, df["mean_entropy"].to_numpy(), width=width, label="Mean H(p_i)")
    ax.bar(x + width, df["expected_kl_to_mean"].to_numpy(), width=width, label="Mean KL(p_i || group mean)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in df[group_col].tolist()], rotation=45, ha="right")
    ax.set_ylabel("Nats")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig

# ==================================================
# Convenience helpers for latest fairness dictionary
# ==================================================

def get_classical_and_uncertainty_matrices(
    fairness: Mapping[str, Any],
    *,
    uncertainty_keys: Sequence[str] = ("predictive_entropy", "aleatoric_entropy", "epistemic_uncertainty"),
) -> Dict[str, np.ndarray]:
    """Extract pairwise matrices from the latest fairness dictionary."""
    matrices: Dict[str, np.ndarray] = {}

    for key in ["statistical_parity", "equal_opportunity", "equalized_odds", "differential_fairness"]:
        if key in fairness and fairness[key] is not None:
            matrices[key] = _to_numpy(fairness[key]["matrix"])

    for key in uncertainty_keys:
        item = fairness.get("uncertainty", {}).get(key)
        if item is not None:
            matrices[key] = _to_numpy(item["pairwise_matrix"])

    return matrices
# Shared condition/metric plotting API. CLI lives in scripts.plot_uncertainty_bars.
from modules.utils.plots.plot_uncertainty_bars import (
    plot_grouped_bars, summarize_metrics, scale_summary,
)
