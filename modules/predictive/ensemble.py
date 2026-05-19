from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


from modules.utils.tensor_utils import DeviceLike, _as_tensor, _default_device
from .metrics import entropy, logits_to_probs

@torch.no_grad()
def evaluate_ensemble(
    models: Sequence[nn.Module],
    loader: torch.utils.data.DataLoader,
    *,
    binary: bool,
    device: Optional[DeviceLike] = None, 
) -> dict[str, np.ndarray]:
    """
    Evaluate an ensemble for binary or multiclass classification.

    Returns arrays with:
        mean_probs: [N, C]
        predictions: [N]
        predictive_entropy: [N]
        aleatoric_uncertainty: [N]
        epistemic_uncertainty: [N]
    """
    if len(models) == 0:
        raise ValueError("At least one model is required for ensemble evaluation.")
    
    device_t = torch.device(device if device is not None else _default_device())
    all_models_probs: list[torch.Tensor] = []

    for model in models:
        model = model.to(device_t)
        model.eval()

        model_probs: list[torch.Tensor] = []
        for batch in loader:
            if len(batch) < 1:
                raise ValueError("Expected batch to contain at least features.")
            x = _as_tensor(batch[0], dtype=torch.float32, device=device_t, name="x_batch")
            logits = model(x)
            probs = logits_to_probs(logits, binary=binary)
            model_probs.append(probs.detach().cpu())

        if len(model_probs) == 0:
            raise ValueError("No batches found in the dataloader; cannot evaluate ensemble.")
        
        all_models_probs.append(torch.cat(model_probs, dim=0))

    # [M, N, C] where M is number of models, N is number of samples, C is number of classes
    ensemble_probs = torch.stack(all_models_probs, dim=0)
    print(f"Ensemble probabilities shape: {ensemble_probs.shape}")
    mean_probs = ensemble_probs.mean(dim=0)
    print(f"Mean probabilities shape: {mean_probs.shape}")

    predictive_entropy = entropy(mean_probs)
    print(f"Predictive entropy shape: {predictive_entropy.shape}")
    aleatoric_uncertainty = entropy(ensemble_probs).mean(dim=0)
    print(f"Aleatoric uncertainty shape: {aleatoric_uncertainty.shape}")
    epistemic_uncertainty = predictive_entropy -  aleatoric_uncertainty
    predictions = mean_probs.argmax(dim=1)

    return {
        "ensemble_probs": ensemble_probs.numpy(),
        "mean_probs": mean_probs.numpy(),
        "predictions": predictions.numpy(),
        "predictive_entropy": predictive_entropy.numpy(),
        "aleatoric_uncertainty": aleatoric_uncertainty.numpy(),
        "epistemic_uncertainty": epistemic_uncertainty.numpy(),
    }