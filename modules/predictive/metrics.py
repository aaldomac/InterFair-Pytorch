from __future__ import annotations

import torch

def logits_to_probs(logits: torch.Tensor, *, binary: bool) -> torch.Tensor:
    """
    Convert logits to probabilities.

    Returns:
        Binary BCE case: `[B, 2]` probabilities for the positive class (after sigmoid).
        Multi-class case: `[B, num_classes]` probabilities (after softmax).
    """
    if binary:
        if logits.ndim == 2 and logits.shape[1] == 1:
            p1 = torch.sigmoid(logits).squeeze(1)
        elif logits.ndim == 1:
            p1 = torch.sigmoid(logits)
        else:
            raise ValueError(f"Expected binary logits of shape (B,) or (B, 1), got {tuple(logits.shape)}")
        return torch.stack([1.0 - p1, p1], dim=1)
    
    if logits.ndim != 2:
        raise ValueError(f"Expected multi-class logits of shape (B, num_classes), got {tuple(logits.shape)}")
    return torch.softmax(logits, dim=1)

def probs_to_labels(probs: torch.Tensor, *, threshold: float = 0.5, binary: bool) -> torch.Tensor:
    """
    Convert probabilities to predicted class labels.

    For binary classification, returns 0 or 1 based on the threshold.
    For multi-class classification, returns the index of the max probability.
    """
    if binary:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        if probs.ndim != 2 or probs.shape[1] != 2:
            raise ValueError(f"Expected binary probabilities of shape (B, 2), got {tuple(probs.shape)}")
        return (probs[:, 1] >= threshold).long()
    
    if probs.ndim != 2:
        raise ValueError(f"Expected multi-class probabilities of shape (B, num_classes), got {tuple(probs.shape)}")
    return torch.argmax(probs, dim=1)

def accuracy_from_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        binary: bool,
        threshold: float = 0.5,
) -> float:
    probs = logits_to_probs(logits, binary=binary)
    preds = probs_to_labels(probs, threshold=threshold, binary=binary)
    return float((preds == targets.long().view(-1)).float().mean().item())

def entropy(probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)