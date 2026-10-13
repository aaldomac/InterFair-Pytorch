import numpy as np
import torch

def expected_calibration_error(preds: torch.Tensor, labels: torch.Tensor, n_bins: int=10) -> float:
    """
    Compute the Expected Calibration Error (ECE) for a multi-class classification model.
    preds should be of shape (batch_size, num_classes) and represent predicted probabilities.
    labels should be of shape (batch_size,) and represent the true class indices.
    """
    if preds.ndim != 2 or preds.shape[0] == 0 or labels.numel() != preds.shape[0]:
        raise ValueError('Expected nonempty, row-aligned probabilities and labels')
    if type(n_bins) is not int or n_bins < 1:
        raise ValueError('n_bins must be a positive integer')
    if not torch.isfinite(preds).all() or (preds < 0).any() or (preds > 1).any():
        raise ValueError('Expected finite probabilities in [0, 1]')
    labels = labels.to(device=preds.device).view(-1)
    # Get predicted probabilities and predicted classes
    # predicted_probs = torch.softmax(preds, dim=1)
    predicted_classes = torch.argmax(preds, dim=1)
    
    # Get confidence scores for the predicted classes
    confidence_scores = preds[torch.arange(preds.size(0), device=preds.device), predicted_classes]
    
    # Create bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=preds.device)
    ece = 0.0
    
    for i in range(n_bins):
        # Get indices of samples in the current bin
        in_bin = (confidence_scores >= bin_boundaries[i]) & (confidence_scores < bin_boundaries[i + 1])
        if i == n_bins - 1:
            in_bin |= confidence_scores == 1.0
        if in_bin.sum() > 0:
            # Calculate accuracy and average confidence for the current bin
            accuracy_in_bin = (predicted_classes[in_bin] == labels[in_bin]).float().mean().item()
            avg_confidence_in_bin = confidence_scores[in_bin].mean().item()
            # Update ECE
            ece += (in_bin.float().mean().item()) * abs(accuracy_in_bin - avg_confidence_in_bin)
    
    return ece