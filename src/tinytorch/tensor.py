"""The :class:`Tensor` -- storage plus a slot in the computation graph.

A ``Tensor`` is deliberately thin.  It owns

* ``data``      -- an ``np.ndarray``, the only place numbers actually live,
* ``requires_grad`` -- whether this tensor is a differentiation target,
* ``grad``      -- accumulated ``dL/dself`` (an ``np.ndarray``, not a Tensor),
* ``grad_fn``   -- the :class:`~tinytorch.autograd.Node` that produced it, or
  ``None`` if it is a **leaf** (user-created, e.g. a parameter).

Everything else -- the operator overloads -- forwards to :mod:`tinytorch.ops`,
which is where the differentiable primitives live.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from . import autograd
from .autograd import Node

__all__ = ["Tensor", "get_default_dtype", "set_default_dtype", "tensor"]

ArrayLike = Any
Shape = tuple[int, ...]

_DEFAULT_DTYPE: np.dtype = np.dtype(np.float32)


def get_default_dtype() -> np.dtype:
    """Floating dtype used when none is supplied (``float32``, like PyTorch)."""
    return _DEFAULT_DTYPE


def set_default_dtype(dtype: Any) -> None:
    """Set the default floating dtype.

    Use ``float64`` when you care about finite-difference gradient checking:
    ``float32`` has ~7 decimal digits, which is not enough to survive the
    catastrophic cancellation in ``(f(x+h) - f(x-h)) / 2h``.
    """
    global _DEFAULT_DTYPE
    dt = np.dtype(dtype)
    if dt.kind != "f":
        raise TypeError(f"default dtype must be a floating type, got {dt}")
    _DEFAULT_DTYPE = dt


def _coerce(data: ArrayLike, dtype: Any | None, copy: bool) -> np.ndarray:
    if isinstance(data, Tensor):
        arr = data.data
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        arr = np.asarray(data)

    if dtype is not None:
        return arr.astype(np.dtype(dtype), copy=copy)
    if arr.dtype.kind in "fc":
        # Keep an explicit float64 array at float64; promote float16 etc. only
        # if the user asked. Python floats land here as float64 -> narrow them
        # to the default dtype so `Tensor([1.0])` matches `torch.tensor([1.0])`.
        if arr.dtype == np.float64 and not isinstance(data, np.ndarray):
            return arr.astype(_DEFAULT_DTYPE, copy=False)
        return arr.copy() if copy else arr
    if arr.dtype.kind in "iub":
        return arr.copy() if copy else arr
    raise TypeError(f"unsupported dtype {arr.dtype} for Tensor")


class Tensor:
    """An n-dimensional array that can remember how it was computed.

    Parameters
    ----------
    data:
        Anything ``np.asarray`` accepts, or another ``Tensor``.
    requires_grad:
        Mark this tensor as a differentiation target.  Only floating tensors
        may require grad.
    dtype:
        Force a dtype.  Defaults to preserving an incoming ndarray's dtype, or
        :func:`get_default_dtype` for Python scalars/lists.
    """

    __slots__ = ("__weakref__", "_requires_grad", "data", "grad", "grad_fn", "retains_grad")

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        dtype: Any | None = None,
        _copy: bool = True,
    ) -> None:
        self.data: np.ndarray = _coerce(data, dtype, _copy)
        self.grad: np.ndarray | None = None
        self.grad_fn: Node | None = None
        self.retains_grad: bool = False
        self._requires_grad = False
        self.requires_grad = requires_grad

    # -- basic properties ---------------------------------------------------
    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        value = bool(value)
        if value and self.data.dtype.kind != "f":
            raise RuntimeError(
                f"only floating tensors can require grad, got dtype {self.data.dtype}"
            )
        if not value and self.grad_fn is not None:
            raise RuntimeError(
                "cannot clear requires_grad on a non-leaf tensor; use .detach()"
            )
        self._requires_grad = value

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> Tensor:
        return self.transpose()

    @property
    def is_leaf(self) -> bool:
        """True if no recorded op produced this tensor (parameters, inputs)."""
        return self.grad_fn is None

    def __len__(self) -> int:
        if self.data.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.data.shape[0]

    def __iter__(self) -> Iterable[Tensor]:
        for i in range(len(self)):
            yield self[i]

    # -- graph interaction --------------------------------------------------
    def backward(self, gradient: ArrayLike | None = None) -> None:
        """Run reverse-mode AD from this tensor. See :func:`autograd.backward`."""
        grad = None
        if gradient is not None:
            grad = gradient.data if isinstance(gradient, Tensor) else np.asarray(gradient)
        autograd.backward(self, grad)

    def _accumulate_grad(self, grad: np.ndarray) -> None:
        """Add *grad* into ``.grad`` (the sum is the point -- see backward())."""
        grad = grad.astype(self.data.dtype, copy=False)
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad = self.grad + grad

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Drop accumulated gradient.

        ``set_to_none=True`` (the default, matching modern PyTorch) releases the
        buffer entirely, which both saves memory and turns "I forgot to call
        backward" into an obvious ``None`` rather than a silent zero update.
        """
        if set_to_none:
            self.grad = None
        elif self.grad is not None:
            self.grad = np.zeros_like(self.grad)

    def detach(self) -> Tensor:
        """Return a tensor sharing the same storage but cut out of the graph.

        The result is a leaf with ``requires_grad=False``.  Storage is
        *shared*, so in-place edits are visible to both -- same semantics as
        ``torch.Tensor.detach``.
        """
        return Tensor(self.data, requires_grad=False, _copy=False)

    def retain_grad(self) -> Tensor:
        """Also accumulate ``.grad`` on this non-leaf tensor (debugging aid)."""
        if self.requires_grad:
            self.retains_grad = True
        return self

    def requires_grad_(self, value: bool = True) -> Tensor:
        self.requires_grad = value
        return self

    # -- conversion ---------------------------------------------------------
    def numpy(self) -> np.ndarray:
        """Return the underlying array (a view, not a copy)."""
        return self.data

    def tolist(self) -> Any:
        return self.data.tolist()

    def item(self) -> float:
        if self.data.size != 1:
            raise ValueError(f"item() requires a single-element tensor, got shape {self.shape}")
        return self.data.reshape(()).item()

    def clone(self) -> Tensor:
        """Differentiable copy: gradient flows straight through."""
        from . import ops

        return ops.clone(self)

    def astype(self, dtype: Any) -> Tensor:
        """Non-differentiable dtype cast (detached)."""
        return Tensor(self.data.astype(np.dtype(dtype)), requires_grad=False)

    def copy_(self, other: Tensor | np.ndarray) -> Tensor:
        """In-place storage overwrite (used by ``load_state_dict``)."""
        src = other.data if isinstance(other, Tensor) else np.asarray(other)
        if src.shape != self.data.shape:
            raise ValueError(f"shape mismatch: {src.shape} into {self.data.shape}")
        self.data[...] = src.astype(self.data.dtype, copy=False)
        return self

    # -- arithmetic ---------------------------------------------------------
    # Every overload defers to tinytorch.ops so that the differentiation rules
    # live in exactly one place.  Import is function-local because ops.py
    # imports this module.
    def __add__(self, other: Any) -> Tensor:
        from . import ops

        return ops.add(self, other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Tensor:
        from . import ops

        return ops.sub(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        from . import ops

        return ops.sub(other, self)

    def __mul__(self, other: Any) -> Tensor:
        from . import ops

        return ops.mul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Tensor:
        from . import ops

        return ops.div(self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        from . import ops

        return ops.div(other, self)

    def __pow__(self, other: Any) -> Tensor:
        from . import ops

        return ops.pow(self, other)

    def __rpow__(self, other: Any) -> Tensor:
        from . import ops

        return ops.pow(other, self)

    def __neg__(self) -> Tensor:
        from . import ops

        return ops.neg(self)

    def __matmul__(self, other: Any) -> Tensor:
        from . import ops

        return ops.matmul(self, other)

    def __rmatmul__(self, other: Any) -> Tensor:
        from . import ops

        return ops.matmul(other, self)

    def __getitem__(self, index: Any) -> Tensor:
        from . import ops

        return ops.getitem(self, index)

    # -- method forms of the ops -------------------------------------------
    def sum(self, axis: Any = None, keepdims: bool = False) -> Tensor:
        from . import ops

        return ops.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Any = None, keepdims: bool = False) -> Tensor:
        from . import ops

        return ops.mean(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: Any) -> Tensor:
        from . import ops

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return ops.reshape(self, shape)

    def view(self, *shape: Any) -> Tensor:
        return self.reshape(*shape)

    def flatten(self, start_dim: int = 0) -> Tensor:
        head = self.shape[:start_dim]
        return self.reshape(*head, -1)

    def transpose(self, *axes: Any) -> Tensor:
        from . import ops

        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return ops.transpose(self, axes if axes else None)

    def permute(self, *axes: Any) -> Tensor:
        return self.transpose(*axes)

    def exp(self) -> Tensor:
        from . import ops

        return ops.exp(self)

    def log(self) -> Tensor:
        from . import ops

        return ops.log(self)

    def sqrt(self) -> Tensor:
        return self ** 0.5

    def abs(self) -> Tensor:
        from . import ops

        return ops.abs(self)

    def maximum(self, other: Any) -> Tensor:
        from . import ops

        return ops.maximum(self, other)

    def minimum(self, other: Any) -> Tensor:
        from . import ops

        return ops.minimum(self, other)

    def relu(self) -> Tensor:
        from . import ops

        return ops.relu(self)

    def sigmoid(self) -> Tensor:
        from . import ops

        return ops.sigmoid(self)

    def tanh(self) -> Tensor:
        from . import ops

        return ops.tanh(self)

    def softmax(self, axis: int = -1) -> Tensor:
        from . import ops

        return ops.softmax(self, axis=axis)

    def log_softmax(self, axis: int = -1) -> Tensor:
        from . import ops

        return ops.log_softmax(self, axis=axis)

    # -- non-differentiable helpers ----------------------------------------
    def argmax(self, axis: Any = None) -> Tensor:
        return Tensor(np.argmax(self.data, axis=axis))

    def max(self, axis: Any = None, keepdims: bool = False) -> Tensor:
        from . import ops

        return ops.max(self, axis=axis, keepdims=keepdims)

    # -- comparisons return plain boolean tensors (no gradient) -------------
    def __eq__(self, other: Any) -> Tensor:  # type: ignore[override]
        return Tensor(self.data == _raw(other))

    def __ne__(self, other: Any) -> Tensor:  # type: ignore[override]
        return Tensor(self.data != _raw(other))

    def __lt__(self, other: Any) -> Tensor:
        return Tensor(self.data < _raw(other))

    def __le__(self, other: Any) -> Tensor:
        return Tensor(self.data <= _raw(other))

    def __gt__(self, other: Any) -> Tensor:
        return Tensor(self.data > _raw(other))

    def __ge__(self, other: Any) -> Tensor:
        return Tensor(self.data >= _raw(other))

    __hash__ = object.__hash__

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        arr = self.data if dtype is None else self.data.astype(dtype)
        return arr.copy() if copy else arr

    def __float__(self) -> float:
        return float(self.item())

    def __bool__(self) -> bool:
        if self.data.size != 1:
            raise RuntimeError(
                "truth value of a multi-element tensor is ambiguous; "
                "use .numpy().any() / .all()"
            )
        return bool(self.data.reshape(()).item())

    def __repr__(self) -> str:
        body = np.array2string(self.data, precision=4, separator=", ", prefix="Tensor(")
        bits = [f"Tensor({body}"]
        if self.grad_fn is not None:
            bits.append(f"grad_fn=<{self.grad_fn.name}>")
        elif self.requires_grad:
            bits.append("requires_grad=True")
        if self.data.dtype != get_default_dtype():
            bits.append(f"dtype={self.data.dtype}")
        return ", ".join(bits) + ")"


def _raw(obj: Any) -> Any:
    return obj.data if isinstance(obj, Tensor) else obj


def tensor(data: ArrayLike, requires_grad: bool = False, dtype: Any | None = None) -> Tensor:
    """Functional alias for :class:`Tensor` (mirrors ``torch.tensor``)."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)
