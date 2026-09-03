"""Tensor factory functions and the global RNG.

All randomness in TinyTorch flows through one ``np.random.Generator`` so that
:func:`manual_seed` makes an entire run -- initialisation, shuffling, dropout
masks -- bit-for-bit reproducible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .tensor import Tensor, get_default_dtype

__all__ = [
    "arange",
    "default_generator",
    "empty",
    "eye",
    "from_numpy",
    "full",
    "linspace",
    "manual_seed",
    "ones",
    "ones_like",
    "rand",
    "randn",
    "zeros",
    "zeros_like",
]

_generator = np.random.default_rng(0)


def manual_seed(seed: int) -> np.random.Generator:
    """Reseed the global generator and return it (mirrors ``torch.manual_seed``)."""
    global _generator
    _generator = np.random.default_rng(seed)
    return _generator


def default_generator() -> np.random.Generator:
    """The generator every TinyTorch random op draws from."""
    return _generator


def _shape(args: tuple[Any, ...]) -> tuple[int, ...]:
    """Accept both ``zeros(2, 3)`` and ``zeros((2, 3))``."""
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(args[0])
    return tuple(int(a) for a in args)


def _dtype(dtype: Any) -> np.dtype:
    return get_default_dtype() if dtype is None else np.dtype(dtype)


def zeros(*shape: Any, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    """Tensor of zeros."""
    return Tensor(np.zeros(_shape(shape), dtype=_dtype(dtype)), requires_grad, _copy=False)


def ones(*shape: Any, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    """Tensor of ones."""
    return Tensor(np.ones(_shape(shape), dtype=_dtype(dtype)), requires_grad, _copy=False)


def full(shape: Any, value: float, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    """Tensor filled with *value*."""
    return Tensor(np.full(_shape((shape,)), value, dtype=_dtype(dtype)), requires_grad, _copy=False)


def empty(*shape: Any, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    """Uninitialised tensor (contents are whatever was in memory)."""
    return Tensor(np.empty(_shape(shape), dtype=_dtype(dtype)), requires_grad, _copy=False)


def eye(n: int, m: int | None = None, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    """Identity matrix."""
    return Tensor(np.eye(n, m, dtype=_dtype(dtype)), requires_grad, _copy=False)


def arange(*args: Any, dtype: Any = None, requires_grad: bool = False) -> Tensor:
    """``np.arange`` with TinyTorch's default dtype."""
    return Tensor(np.arange(*args, dtype=_dtype(dtype)), requires_grad, _copy=False)


def linspace(start: float, stop: float, num: int = 50, dtype: Any = None,
             requires_grad: bool = False) -> Tensor:
    """Evenly spaced values over ``[start, stop]``."""
    return Tensor(np.linspace(start, stop, num, dtype=_dtype(dtype)), requires_grad, _copy=False)


def rand(*shape: Any, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    """Uniform samples on ``[0, 1)``."""
    arr = _generator.random(_shape(shape), dtype=np.float64).astype(_dtype(dtype))
    return Tensor(arr, requires_grad, _copy=False)


def randn(*shape: Any, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    """Standard normal samples."""
    arr = _generator.standard_normal(_shape(shape)).astype(_dtype(dtype))
    return Tensor(arr, requires_grad, _copy=False)


def zeros_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """Zeros with the shape and dtype of *t*."""
    return Tensor(np.zeros_like(t.data), requires_grad, _copy=False)


def ones_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """Ones with the shape and dtype of *t*."""
    return Tensor(np.ones_like(t.data), requires_grad, _copy=False)


def from_numpy(array: np.ndarray, requires_grad: bool = False) -> Tensor:
    """Wrap an existing array *without copying* -- the two share storage."""
    return Tensor(array, requires_grad, _copy=False)
