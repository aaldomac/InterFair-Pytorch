from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch


ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[int], Sequence[float]]
DeviceLike = Union[torch.device, str]

def _as_tensor(
    x: ArrayLike,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLike] = None,
    name: str = "tensor",
    detach: bool = False,
) -> torch.Tensor:
    """
    Convert input to a tensor, optionally moving it to a device and dtype.

    This function is intentionally permissive and should be used by generic
    utilities and metrics. For model training code, pass dtype and device
    explicitly.

    Args:
        x: Tensor-like input.
        dtype: Optional target dtype.
        device: Optional target device.
        name: Name used in error messages.
        detach: Whether to detach an existing tensor.

    Returns:
        Tensor with requested dtype/device.
    """
    try:
        if torch.is_tensor(x):
            tensor = x.detach() if detach else x
            if dtype is not None or device is not None:
                tensor = tensor.to(device=device, dtype=dtype)
            return tensor

        return torch.as_tensor(x, dtype=dtype, device=device)

    except Exception as exc:
        raise TypeError(f"Could not convert {name} to a tensor.") from exc
    
def _as_1d_tensor(
    x: ArrayLike,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLike] = None,
    name: str = "tensor",
    detach: bool = True,
) -> torch.Tensor:
    """
    Convert input to a flattened 1D tensor.

    Useful for labels, predictions, group IDs, and scalar uncertainty vectors.
    """
    tensor = _as_tensor(x, dtype=dtype, device=device, name=name, detach=detach).view(-1)

    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after flattening, got {tuple(tensor.shape)}.")

    return tensor

def _to_long_tensor(x: Any, device: Optional[DeviceLike] = None) -> torch.Tensor:
    return _as_tensor(x, dtype=torch.long, device=device, name="long tensor")


def _to_float_tensor(x: Any, device: Optional[DeviceLike] = None) -> torch.Tensor:
    return _as_tensor(x, dtype=torch.float32, device=device, name="float tensor")

def _default_device() -> torch.device:
    """
    Get the default device (GPU if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================
# Generic validation helpers
# ==================================================
def _validate_same_length(**tensors: torch.Tensor) -> None:
    """
    Validate that all tensors have the same first/flattened length.

    Example:
        _validate_same_length(preds=preds, labels=labels, group_ids=group_ids)
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided.")

    lengths = {name: int(t.view(-1).shape[0]) for name, t in tensors.items()}

    if len(set(lengths.values())) != 1:
        raise ValueError(f"All tensors must have the same length. Got: {lengths}")


# ==================================================
# Small tensor utilities useful across metrics/models
# ==================================================
def _unique_sorted_long(x: ArrayLike, *, name: str = "values") -> torch.Tensor:
    """Return sorted unique long values from a 1D tensor-like input."""
    x_t = _as_1d_tensor(x, dtype=torch.long, name=name)
    return torch.unique(x_t, sorted=True)
