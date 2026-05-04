import numpy as np
import torch


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the accuracy of predictions.
    preds should be of shape (batch_size, num_classes) and represent predicted probabilities or logits.
    labels should be of shape (batch_size,) and represent the true class indices.
    """
    predicted_classes = torch.argmax(preds, dim=1)
    correct = (predicted_classes == labels).sum().item()
    total = labels.size(0)
    return correct / total

def true_positive_rate(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the true positive rate (TPR) for binary classification.
    preds should be of shape (batch_size,) and represent predicted probabilities for the positive class.
    labels should be of shape (batch_size,) and represent the true binary labels (0 or 1).
    """
    predicted_positive = (preds >= 0.5).float()
    true_positive = ((predicted_positive == 1) & (labels == 1)).sum().item()
    actual_positive = (labels == 1).sum().item()
    
    if actual_positive == 0:
        return 0.0
    return true_positive / actual_positive

def false_positive_rate(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the false positive rate (FPR) for binary classification.
    preds should be of shape (batch_size,) and represent predicted probabilities for the positive class.
    labels should be of shape (batch_size,) and represent the true binary labels (0 or 1).
    """
    predicted_positive = (preds >= 0.5).float()
    false_positive = ((predicted_positive == 1) & (labels == 0)).sum().item()
    actual_negative = (labels == 0).sum().item()
    
    if actual_negative == 0:
        return 0.0
    return false_positive / actual_negative

def kl_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float=1e-10) -> torch.Tensor:
    """
    Compute the KL divergence between two distributions p and q.
    Both p and q should be of shape (batch_size, num_classes) and represent probability distributions (i.e., sum to 1).
    """
    # Add a small epsilon to avoid log(0)
    p = p + epsilon
    q = q + epsilon
    
    kl_div = torch.sum(p * torch.log(p / q), dim=1)
    return kl_div

def cross_entropy(p: torch.Tensor, q: torch.Tensor, epsilon: float=1e-10) -> torch.Tensor:
    """
    Compute the cross-entropy for two distributions p and q.
    Both p and q should be of shape (batch_size, num_classes) and represent probability distributions (i.e., sum to 1).
    """
    log_q = torch.log(q + epsilon)  # Add epsilon to avoid log(0)
    ce = -torch.sum(p * log_q, dim=1)
    return ce