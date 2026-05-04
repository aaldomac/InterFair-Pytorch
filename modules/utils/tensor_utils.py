import torch
import numpy as np

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[int], Sequence[float]]
DeviceLike = Union[torch.device, str]

def _as_tensor(
    x: ArrayLike,
    *,
    dtype: torch.dtype,
    device: DeviceLike,
    name: str
) -> torch.Tensor:
    """
    Convert input to a tensor on the requested device and type.
    """
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    try:
        return torch.as_tensor(x, dtype=dtype, device=device)
    except Exception as exc:
        raise TypeError(f"Could not convert {name} to a tensor.") from exc

def _to_long_tensor(x: Any, device: Optional[DeviceLike] = None) -> torch.Tensor:
    return _as_tensor(x, dtype=torch.long, device=device, name="long tensor")

def _to_float_tensor(x: Any, device: Optional[DeviceLike] = None) -> torch.Tensor:
    return _as_tensor(x, dtype=torch.float32, device=device, name="float tensor")

def _validate_vocab_sizes(vocab_sizes: Sequence[int]) -> List[int]:
    """
    Validate categorical vocabulary sizes.
    """
    if not isinstance(vocab_sizes, Sequence) or len(vocab_sizes) == 0:
        raise ValueError("vocab_sizes must be anon-empty sequence of positive integers.")

    vocab_sizes = list(vocab_sizes)
    if any((not isinstance(v, int)) or v<=0 for v in vocab_sizes):
        raise ValueError("All vocab sizes must be positive integers.")
    return vocab_sizes

def _validate_2d_long_tensor(x: torch.Tensor, *, expected_second_dim: Optional[int], name: str) -> None:
    """
    Validate that x is a 2D integer tensor with optional second dimension check.
    """
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [B, D], got {tuple(x.shape)}.")
    if expected_second_dim is not None and x.shape[1] != expected_second_dim:
        raise ValueError(f"{name} must have shape [B, {expected_second_dim}], got {tuple(x.shape)}.")
    if x.dtype != torch.long:
        raise TypeError(f"{name} must be of dtype torch.long, got {x.dtype}.")

def _validate_1d_long_tensor(x: torch.Tensor, *, name:str) -> None:
    """
    Validate that x is a 1D integer tensor.
    """
    if x.ndim != 1:
        raise ValueError(f"{name} must have shape [B], got {tuple(x.shape)}.")
    if x.dtype != torch.long:
        raise TypeError(f"{name} must be of dtype torch.long, got {x.dtype}.")

def _validate_2d_float_tensor(x: torch.Tensor, *, expected_second_dim: Optional[int], name: str) -> None:
    """
    Validate that x is a 2D float tensor with optional second dimension check.
    """
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [B, D], got {tuple(x.shape)}.")
    if expected_second_dim is not None and x.shape[1] != expected_second_dim:
        raise ValueError(f"{name} must have shape [B, {expected_second_dim}], got {tuple(x.shape)}.")
    if not x.dtype.is_floating_point:
        raise TypeError(f"{name} must be of floating point dtype, got {x.dtype}.")

def _default_device() -> torch.device:
    """
    Get the default device (GPU if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")