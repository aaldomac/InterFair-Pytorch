import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

def plot_fairness_matrix(fairness_matrix, group_names, metric_name):
    """
    Plots a fairness matrix as a heatmap.

    Parameters:
    - fairness_matrix: A 2D numpy array representing the fairness metrics between groups.
    - group_names: A list of group names corresponding to the rows and columns of the matrix.
    - metric_name: The name of the fairness metric being plotted (e.g., "Demographic Parity").

    Returns:
    - A matplotlib figure object containing the heatmap.
    """
    if hasattr(fairness_matrix, "detach"):
        fairness_matrix = fairness_matrix.detach().cpu().numpy()
    if hasattr(group_names, "detach"):
        group_names = group_names.detach().cpu().numpy()

    group_names = [str(g) for g in group_names]
    n = len(group_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a heatmap
    cax = ax.matshow(fairness_matrix, cmap='viridis')
    
    # Set axis labels
    # ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    # ax.set_xticklabels(group_names)
    ax.set_yticklabels(group_names)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add color bar
    fig.colorbar(cax)
    
    # Set titles
    ax.set_xlabel("Group")
    ax.set_ylabel("Group")
    plt.title(f"{metric_name} Fairness Matrix")
    
    return fig


def mix_fairness_matrices(matrix_a, matrix_b):
    """
    Mixes two symmetric fairness matrices by showing in the upper triangle the values from matrix_a and in the lower triangle the values from matrix_b.
    The diagonal will be set to NaN, 0 or 1.

    Parameters:
    - matrix_a: A 2D numpy array representing the first fairness matrix.
    - matrix_b: A 2D numpy array representing the second fairness matrix.
    Returns:
    - A 2D numpy array representing the mixed fairness matrix.
    """
    assert matrix_a.shape == matrix_b.shape, "Both matrices must have the same shape."
    assert matrix_a.shape[0] == matrix_a.shape[1], "Matrices must be square."
    
    mixed_matrix = np.zeros_like(matrix_a)
    
    # Fill upper triangle with values from matrix_a
    mixed_matrix[np.triu_indices_from(mixed_matrix, k=1)] = matrix_a[np.triu_indices_from(matrix_a, k=1)]
    
    # Fill lower triangle with values from matrix_b
    mixed_matrix[np.tril_indices_from(mixed_matrix, k=-1)] = matrix_b[np.tril_indices_from(matrix_b, k=-1)]
    
    # Set diagonal to NaN (or 0 or 1 depending on the context)
    np.fill_diagonal(mixed_matrix, np.nan)  # or 0 or 1

    # Normalize upper and lower triangles (to their respective max values) for better visualization
    upper_max = np.nanmax(matrix_a)
    lower_max = np.nanmax(matrix_b)
    if upper_max > 0:
        mixed_matrix[np.triu_indices_from(mixed_matrix, k=1)] /= upper_max
    if lower_max > 0:
        mixed_matrix[np.tril_indices_from(mixed_matrix, k=-1)] /= lower_max
    
    return mixed_matrix

# From the mixed matrix, scatter plot the values of the upper triangle ordered increasingly and the values of the lower triangle with the order of the upper triangle, to see if there is a correlation between the two metrics.
try:
    from scipy.stats import spearmanr, kendalltau, pearsonr
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    
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
    Compare two square fairness matrices through their pairwise entries and
    visualize ordering correlation.

    Parameters
    ----------
    matrix_a : np.ndarray or torch.Tensor
        First square matrix.
    matrix_b : np.ndarray or torch.Tensor
        Second square matrix.
    correlation : str, default="spearman"
        One of:
            - "spearman": rank correlation
            - "kendall": Kendall tau rank correlation
            - "pearson": linear correlation on raw values
    plot_mode : str, default="ranks"
        What to visualize:
            - "ranks": scatter rank_a vs rank_b for each pair
            - "sorted_values": sort by metric A and plot both sequences
            - "order_positions": compare permutation/order positions
    triangle : str, default="upper"
        Which triangle to extract:
            - "upper"
            - "lower"
    normalize : bool, default=False
        Whether to normalize the extracted values into [0,1] independently.
        Only affects value-based plots, not ranks.
    labels : tuple[str, str], default=("Metric A", "Metric B")
        Labels for legend and axes.
    return_data : bool, default=False
        If True, also return a dictionary with extracted values, ranks, etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
    stats : dict, optional
        Returned only if return_data=True.
    """
    # -----------------------------
    # Convert to numpy
    # -----------------------------
    if hasattr(matrix_a, "detach"):
        matrix_a = matrix_a.detach().cpu().numpy()
    else:
        matrix_a = np.asarray(matrix_a)

    if hasattr(matrix_b, "detach"):
        matrix_b = matrix_b.detach().cpu().numpy()
    else:
        matrix_b = np.asarray(matrix_b)

    # -----------------------------
    # Checks
    # -----------------------------
    assert matrix_a.shape == matrix_b.shape, "Both matrices must have the same shape."
    assert matrix_a.ndim == 2, "Matrices must be 2D."
    assert matrix_a.shape[0] == matrix_a.shape[1], "Matrices must be square."

    n = matrix_a.shape[0]
    if n < 2:
        raise ValueError("Matrices must be at least 2x2.")

    # -----------------------------
    # Extract corresponding entries
    # -----------------------------
    if triangle == "upper":
        idx = np.triu_indices_from(matrix_a, k=1)
    elif triangle == "lower":
        idx = np.tril_indices_from(matrix_a, k=-1)
    else:
        raise ValueError("triangle must be 'upper' or 'lower'.")

    values_a = matrix_a[idx].astype(float)
    values_b = matrix_b[idx].astype(float)

    # -----------------------------
    # Optional normalization
    # -----------------------------
    if normalize:
        def _safe_normalize(x):
            x_min = np.min(x)
            x_max = np.max(x)
            if np.isclose(x_max, x_min):
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min)

        values_a_plot = _safe_normalize(values_a)
        values_b_plot = _safe_normalize(values_b)
    else:
        values_a_plot = values_a.copy()
        values_b_plot = values_b.copy()

    # -----------------------------
    # Ranks
    # rank 0 = smallest
    # -----------------------------
    order_a = np.argsort(values_a)
    order_b = np.argsort(values_b)

    rank_a = np.empty_like(order_a)
    rank_a[order_a] = np.arange(len(order_a))

    rank_b = np.empty_like(order_b)
    rank_b[order_b] = np.arange(len(order_b))

    # -----------------------------
    # Correlation computation
    # -----------------------------
    correlation = correlation.lower()
    if correlation not in {"spearman", "kendall", "pearson"}:
        raise ValueError("correlation must be 'spearman', 'kendall', or 'pearson'.")

    if correlation == "pearson":
        x_corr = values_a
        y_corr = values_b
    else:
        x_corr = rank_a
        y_corr = rank_b

    if _SCIPY_AVAILABLE:
        if correlation == "spearman":
            corr_value, p_value = spearmanr(values_a, values_b)
        elif correlation == "kendall":
            corr_value, p_value = kendalltau(values_a, values_b)
        else:
            corr_value, p_value = pearsonr(values_a, values_b)
    else:
        # Fallback without scipy
        if correlation == "kendall":
            raise ImportError("scipy is required for Kendall correlation.")
        corr_value = np.corrcoef(x_corr, y_corr)[0, 1]
        p_value = np.nan

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    if plot_mode == "ranks":
        ax.scatter(rank_a, rank_b, alpha=0.7)
        lim = len(rank_a) - 1
        ax.plot([0, lim], [0, lim], linestyle="--", alpha=0.6)
        ax.set_xlabel(f"Rank in {labels[0]}")
        ax.set_ylabel(f"Rank in {labels[1]}")
        ax.set_title(
            f"Ordering comparison ({correlation.capitalize()} = {corr_value:.4f})"
        )

    elif plot_mode == "sorted_values":
        sorted_a = values_a_plot[order_a]
        sorted_b_by_a = values_b_plot[order_a]

        ax.scatter(np.arange(len(sorted_a)), sorted_a, alpha=0.7, label=labels[0])
        ax.scatter(np.arange(len(sorted_b_by_a)), sorted_b_by_a, alpha=0.7, label=labels[1])
        ax.set_xlabel(f"Pair index (sorted by {labels[0]})")
        ax.set_ylabel("Normalized value" if normalize else "Value")
        ax.set_title(
            f"Sorted pairwise values ({correlation.capitalize()} = {corr_value:.4f})"
        )
        ax.legend()

    elif plot_mode == "order_positions":
        ax.scatter(np.arange(len(order_a)), order_a, alpha=0.7, label=labels[0])
        ax.scatter(np.arange(len(order_b)), order_b, alpha=0.7, label=labels[1])
        ax.set_xlabel("Sorted position")
        ax.set_ylabel("Original pair index")
        ax.set_title(
            f"Permutation comparison ({correlation.capitalize()} = {corr_value:.4f})"
        )
        ax.legend()

    else:
        raise ValueError(
            "plot_mode must be 'ranks', 'sorted_values', or 'order_positions'."
        )

    ax.grid(alpha=0.3)

    stats = {
        "correlation_type": correlation,
        "correlation_value": corr_value,
        "p_value": p_value,
        "values_a": values_a,
        "values_b": values_b,
        "rank_a": rank_a,
        "rank_b": rank_b,
        "order_a": order_a,
        "order_b": order_b,
        "triangle": triangle,
    }

    if return_data:
        return fig, stats
    return fig