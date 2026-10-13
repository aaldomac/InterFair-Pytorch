from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


from modules.utils.tensor_utils import DeviceLike, _as_tensor, _default_device
from .metrics import entropy, logits_to_probs, probs_to_labels

@torch.no_grad()
def evaluate_ensemble(
    models: Sequence[nn.Module],
    loader: torch.utils.data.DataLoader,
    *,
    binary: bool,
    device: Optional[DeviceLike] = None, 
    threshold: float = 0.5,
    store_member_probs: bool = True
) -> dict[str, np.ndarray]:
    """Predict each batch once per member, with optional [M,N,C] storage.

    All members see exactly the same input batch. Without member storage,
    retained probability arrays require O(N*C) rather than O(M*N*C) memory.
    Callers aligning outputs to an external table must use an ordered loader.

    Returns arrays with:
        mean_probs: [N, C]
        predictions: [N]
        predictive_entropy: [N]
        aleatoric_uncertainty: [N]
        epistemic_uncertainty: [N]
    """
    if not models:
        raise ValueError('At least one model is required for ensemble evaluation.')
    device_t = torch.device(device if device is not None else _default_device())
    for model in models:
        model.to(device_t).eval()
    means, aleatorics = [], []
    member_batches = [[] for _ in models] if store_member_probs else None
    for batch in loader:
        if not len(batch):
            raise ValueError('Expected batch to contain at least features.')
        x = _as_tensor(batch[0], dtype=torch.float32, device=device_t, name='x_batch')
        prob_sum = entropy_sum = None
        for index, model in enumerate(models):
            probs = logits_to_probs(model(x), binary=binary).detach().cpu()
            member_entropy = entropy(probs)
            prob_sum = probs.clone() if prob_sum is None else prob_sum + probs
            entropy_sum = member_entropy if entropy_sum is None else entropy_sum + member_entropy
            if member_batches is not None:
                member_batches[index].append(probs)
        means.append(prob_sum / len(models))
        aleatorics.append(entropy_sum / len(models))
    if not means:
        raise ValueError('No batches found in the dataloader; cannot evaluate ensemble.')
    mean_probs = torch.cat(means)
    aleatoric = torch.cat(aleatorics)
    total = entropy(mean_probs)
    result = dict(mean_probs=mean_probs.numpy(),
                  predictions=probs_to_labels(mean_probs, binary=binary, threshold=threshold).numpy(),
                  predictive_entropy=total.numpy(), aleatoric_uncertainty=aleatoric.numpy(),
                  epistemic_uncertainty=(total-aleatoric).numpy())
    if member_batches is not None:
        result['ensemble_probs'] = torch.stack([torch.cat(parts) for parts in member_batches]).numpy()
    return result
