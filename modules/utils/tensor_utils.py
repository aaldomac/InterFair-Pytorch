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

def _validate_tensor_keys(tensors: Mapping[str, torch.Tensor], required_keys: Sequence[str]) -> None:
    """Validate that a tensor dictionary contains required keys."""
    missing = [key for key in required_keys if key not in tensors]
    if missing:
        raise KeyError(f"Missing required tensor keys: {missing}")
        
def _validate_vocab_sizes(vocab_sizes: Sequence[int]) -> List[int]:
    """Validate categorical vocabulary sizes."""
    if not isinstance(vocab_sizes, Sequence) or len(vocab_sizes) == 0:
        raise ValueError("vocab_sizes must be a non-empty sequence of positive integers.")

    vocab_sizes = list(vocab_sizes)
    if any((not isinstance(v, int)) or v <= 0 for v in vocab_sizes):
        raise ValueError("All vocab sizes must be positive integers.")

    return vocab_sizes

# ==================================================
# Shape and dtype validation helpers
# ==================================================
def _validate_1d_tensor(
    x: torch.Tensor,
    *,
    name: str,
    dtype: Optional[torch.dtype] = None,
    floating: Optional[bool] = None,
) -> None:
    """
    Validate that x is a 1D tensor, with optional dtype checks.

    Args:
        x: Tensor to validate.
        name: Name used in error messages.
        dtype: Exact dtype required. If provided, overrides `floating`.
        floating: If True, require floating-point dtype. If False, require non-floating dtype.
    """
    if x.ndim != 1:
        raise ValueError(f"{name} must have shape [B], got {tuple(x.shape)}.")

    if dtype is not None and x.dtype != dtype:
        raise TypeError(f"{name} must be of dtype {dtype}, got {x.dtype}.")

    if dtype is None and floating is not None:
        if floating and not x.dtype.is_floating_point:
            raise TypeError(f"{name} must be of floating point dtype, got {x.dtype}.")
        if not floating and x.dtype.is_floating_point:
            raise TypeError(f"{name} must be of non-floating dtype, got {x.dtype}.")


def _validate_2d_tensor(
    x: torch.Tensor,
    *,
    expected_second_dim: Optional[int],
    name: str,
    dtype: Optional[torch.dtype] = None,
    floating: Optional[bool] = None,
) -> None:
    """
    Validate that x is a 2D tensor with optional second-dimension and dtype checks.
    """
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [B, D], got {tuple(x.shape)}.")

    if expected_second_dim is not None and x.shape[1] != expected_second_dim:
        raise ValueError(f"{name} must have shape [B, {expected_second_dim}], got {tuple(x.shape)}.")

    if dtype is not None and x.dtype != dtype:
        raise TypeError(f"{name} must be of dtype {dtype}, got {x.dtype}.")

    if dtype is None and floating is not None:
        if floating and not x.dtype.is_floating_point:
            raise TypeError(f"{name} must be of floating point dtype, got {x.dtype}.")
        if not floating and x.dtype.is_floating_point:
            raise TypeError(f"{name} must be of non-floating dtype, got {x.dtype}.")

def _validate_1d_long_tensor(x: torch.Tensor, *, name: str) -> None:
    """Validate that x is a 1D long tensor."""
    _validate_1d_tensor(x, name=name, dtype=torch.long)


def _validate_2d_long_tensor(
    x: torch.Tensor,
    *,
    expected_second_dim: Optional[int],
    name: str,
) -> None:
    """Validate that x is a 2D long tensor with optional second-dimension check."""
    _validate_2d_tensor(x, expected_second_dim=expected_second_dim, name=name, dtype=torch.long)


def _validate_1d_float_tensor(x: torch.Tensor, *, name: str) -> None:
    """Validate that x is a 1D floating-point tensor."""
    _validate_1d_tensor(x, name=name, floating=True)


def _validate_2d_float_tensor(
    x: torch.Tensor,
    *,
    expected_second_dim: Optional[int],
    name: str,
) -> None:
    """Validate that x is a 2D floating-point tensor with optional second-dimension check."""
    _validate_2d_tensor(x, expected_second_dim=expected_second_dim, name=name, floating=True)

# ==================================================
# Small tensor utilities useful across metrics/models
# ==================================================

def _unique_sorted_long(x: ArrayLike, *, name: str = "values") -> torch.Tensor:
    """Return sorted unique long values from a 1D tensor-like input."""
    x_t = _as_1d_tensor(x, dtype=torch.long, name=name)
    return torch.unique(x_t, sorted=True)


def _safe_mean(x: torch.Tensor, *, empty_value: float = 0.0) -> float:
    """Return mean as float, with a controlled value for empty tensors."""
    if x.numel() == 0:
        return float(empty_value)
    return float(x.float().mean().item())


def _as_numpy(x: Any) -> np.ndarray:
    """Convert tensor-like input to a NumPy array on CPU."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)