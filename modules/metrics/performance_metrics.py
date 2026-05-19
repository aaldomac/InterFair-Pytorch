from __future__ import annotations

from typing import Optional

import torch

from modules.utils.tensor_utils import _as_tensor, _as_1d_tensor, _validate_same_length

# TODO: Consider if the positive_class argument is necessary.
def prediction_scores_to_labels(
    preds: torch.Tensor,
    *,
    threshold: float = 0.5,
    positive_class: int = 1,
) -> torch.Tensor:
    """Convert prediction scores/logits/probabilities/hard labels to class labels.

    Accepted shapes:
        - [N]: binary scores/probabilities or hard labels
        - [N, 1]: binary scores/probabilities
        - [N, C]: class scores/probabilities/logits

    Notes:
        For [N] and [N, 1], values are thresholded.
        For [N, C], argmax is used.
    """
    preds_t = _as_tensor(preds, name="preds")

    if preds_t.ndim == 1:
        # If already integer labels, keep them. Otherwise threshold as binary scores.
        if preds_t.dtype in (torch.long, torch.int64, torch.int32, torch.int16, torch.int8):
            return preds_t.long().view(-1)
        return (preds_t.float().view(-1) >= threshold).long()

    if preds_t.ndim == 2 and preds_t.shape[1] == 1:
        return (preds_t.squeeze(1).float() >= threshold).long()

    if preds_t.ndim == 2:
        return torch.argmax(preds_t, dim=1).long()

    raise ValueError(f"preds must have shape [N], [N, 1], or [N, C], got {tuple(preds_t.shape)}.")


def positive_class_scores(
    preds: torch.Tensor,
    *,
    positive_class: int = 1,
) -> torch.Tensor:
    """Extract positive-class scores from binary or multiclass predictions.

    Accepted shapes:
        - [N]: returned as-is as float
        - [N, 1]: squeezed to [N]
        - [N, C]: column `positive_class`
    """
    preds_t = _as_tensor(preds, name="preds")

    if preds_t.ndim == 1:
        return preds_t.float().view(-1)

    if preds_t.ndim == 2 and preds_t.shape[1] == 1:
        return preds_t.squeeze(1).float().view(-1)

    if preds_t.ndim == 2:
        if not 0 <= positive_class < preds_t.shape[1]:
            raise ValueError(
                f"positive_class must be in [0, {preds_t.shape[1] - 1}], got {positive_class}."
            )
        return preds_t[:, positive_class].float().view(-1)

    raise ValueError(f"preds must have shape [N], [N, 1], or [N, C], got {tuple(preds_t.shape)}.")


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        preds: Either hard labels [N], binary scores [N]/[N, 1], or class scores [N, C].
        labels: True class labels [N].
    """
    labels_t = _as_1d_tensor(labels, dtype=torch.long, name="labels")
    pred_labels = prediction_scores_to_labels(preds)
    _validate_same_length(preds=pred_labels, labels=labels_t)

    if labels_t.numel() == 0:
        raise ValueError("Cannot compute accuracy on empty labels.")

    return float((pred_labels == labels_t).float().mean().item())

# TODO: This function is not OK for multiclass.
def true_positive_rate(
    preds: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
    positive_class: int = 1,
    empty_value: float = 0.0,
) -> float:
    """Compute true positive rate for a chosen positive class.

    For binary classification, `preds` can be probabilities/scores or hard labels.
    For multiclass predictions [N, C], predicted labels are obtained with argmax.
    """
    labels_t = _as_1d_tensor(labels, dtype=torch.long, name="labels")
    pred_labels = prediction_scores_to_labels(
        preds,
        threshold=threshold,
        positive_class=positive_class,
    )
    _validate_same_length(preds=pred_labels, labels=labels_t)

    actual_positive = labels_t == positive_class
    denom = int(actual_positive.sum().item())
    if denom == 0:
        return float(empty_value)

    true_positive = (pred_labels[actual_positive] == positive_class).sum().item()
    return float(true_positive / denom)

def false_positive_rate(
    preds: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
    positive_class: int = 1,
    empty_value: float = 0.0,
) -> float:
    """Compute false positive rate for a chosen positive class.

    FPR = FP / actual negatives, where actual negatives are labels != positive_class.
    """
    labels_t = _as_1d_tensor(labels, dtype=torch.long, name="labels")
    pred_labels = prediction_scores_to_labels(
        preds,
        threshold=threshold,
        positive_class=positive_class,
    )
    _validate_same_length(preds=pred_labels, labels=labels_t)

    actual_negative = labels_t != positive_class
    denom = int(actual_negative.sum().item())
    if denom == 0:
        return float(empty_value)

    false_positive = (pred_labels[actual_negative] == positive_class).sum().item()
    return float(false_positive / denom)

def kl_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Compute KL divergence KL(p || q) for batched distributions.

    Args:
        p: [B, C] probability distributions.
        q: [B, C] probability distributions.

    Returns:
        Tensor of shape [B].
    """
    p_t = _as_tensor(p, dtype=torch.float32, name="p")
    q_t = _as_tensor(q, dtype=torch.float32, name="q")

    if p_t.shape != q_t.shape:
        raise ValueError(f"p and q must have the same shape. Got {tuple(p_t.shape)} and {tuple(q_t.shape)}.")
    if p_t.ndim != 2:
        raise ValueError(f"p and q must have shape [B, C], got {tuple(p_t.shape)}.")

    p_safe = p_t.clamp_min(epsilon)
    q_safe = q_t.clamp_min(epsilon)
    return torch.sum(p_safe * torch.log(p_safe / q_safe), dim=1)


def cross_entropy(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Compute cross-entropy H(p, q) for batched distributions.

    Args:
        p: [B, C] target probability distributions.
        q: [B, C] predicted probability distributions.

    Returns:
        Tensor of shape [B].
    """
    p_t = _as_tensor(p, dtype=torch.float32, name="p")
    q_t = _as_tensor(q, dtype=torch.float32, name="q")

    if p_t.shape != q_t.shape:
        raise ValueError(f"p and q must have the same shape. Got {tuple(p_t.shape)} and {tuple(q_t.shape)}.")
    if p_t.ndim != 2:
        raise ValueError(f"p and q must have shape [B, C], got {tuple(p_t.shape)}.")

    return -torch.sum(p_t * torch.log(q_t.clamp_min(epsilon)), dim=1)


def entropy(p: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Compute categorical entropy over the last dimension.

    Args:
        p: Tensor of probabilities with shape [..., C].

    Returns:
        Tensor of entropy values with shape [...].
    """
    p_t = _as_tensor(p, dtype=torch.float32, name="p")
    return -torch.sum(p_t * torch.log(p_t.clamp_min(epsilon)), dim=-1)