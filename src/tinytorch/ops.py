"""Differentiable primitives.

Each operation is a :class:`~tinytorch.autograd.Function` subclass with a
static ``forward`` (NumPy in, NumPy out) and a static ``backward`` (the
vector-Jacobian product).  Below each class sits a lowercase wrapper that
coerces Python scalars / ndarrays into :class:`~tinytorch.tensor.Tensor` and
calls ``Function.apply``.

Two conventions run through the whole file:

* **Broadcasting is not each op's problem.**  ``Mul.backward`` may return a
  gradient of the *broadcast* shape; the engine calls
  :func:`~tinytorch.autograd.unbroadcast` before accumulating it into the
  input.  One implementation, every op covered.
* **Config is keyword-only.**  Axes, indices and reduction flags never occupy a
  positional slot, so ``backward`` returns exactly one gradient per positional
  input.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .autograd import Context, Function
from .tensor import Tensor

__all__ = [
    "abs",
    "add",
    "clone",
    "div",
    "exp",
    "gelu",
    "getitem",
    "log",
    "log_softmax",
    "matmul",
    "max",
    "maximum",
    "mean",
    "minimum",
    "mul",
    "neg",
    "pow",
    "relu",
    "reshape",
    "sigmoid",
    "softmax",
    "sub",
    "sum",
    "tanh",
    "transpose",
]



def _ensure(x: Any, like: Tensor | None = None) -> Tensor:
    """Wrap scalars/arrays as a Tensor, adopting *like*'s dtype for scalars."""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, np.ndarray):
        return Tensor(x, _copy=False)
    dtype = like.dtype if (like is not None and like.dtype.kind == "f") else None
    return Tensor(x, dtype=dtype)


def _restore_reduced(
    grad: np.ndarray, shape: tuple[int, ...], axis: Any, keepdims: bool
) -> np.ndarray:
    """Broadcast a reduction's gradient back over the axes it collapsed."""
    if not keepdims and axis is not None:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        axes = tuple(a % len(shape) for a in axes)
        grad = np.expand_dims(grad, axes)
    return np.broadcast_to(grad, shape)


# ---------------------------------------------------------------------------
# Elementwise binary ops
# ---------------------------------------------------------------------------
class Add(Function):
    """``y = a + b``.  Addition fans the gradient out unchanged: dy/da = dy/db = 1."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return g, g


class Sub(Function):
    """``y = a - b``; dy/da = 1, dy/db = -1."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return g, -g


class Mul(Function):
    """``y = a * b``; each input's gradient is scaled by the *other* input."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        return g * b, g * a


class Div(Function):
    """``y = a / b``; dy/da = 1/b, dy/db = -a/b^2."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a / b

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        return g / b, -g * a / (b * b)


class Pow(Function):
    """``y = a ** b``.

    ``dy/da = b * a**(b-1)`` always; ``dy/db = y * ln a`` only where ``a > 0``.
    The exponent gradient is masked to zero elsewhere rather than propagating
    NaN -- the derivative genuinely does not exist there, and in the
    overwhelmingly common case (a constant exponent) it is discarded anyway.
    """

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = a ** b
        ctx.save_for_backward(a, b, out)
        return out

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b, out = ctx.saved_tensors
        with np.errstate(divide="ignore", invalid="ignore"):
            grad_a = g * b * (a ** (b - 1))
            log_a = np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), 0.0)
        grad_b = g * out * log_a
        return grad_a, grad_b


class Maximum(Function):
    """Elementwise ``max(a, b)``.

    At a tie the subgradient is not unique; we split it 0.5/0.5, matching
    ``torch.maximum``.  (``relu`` uses a different convention -- see :class:`ReLU`.)
    """

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return np.maximum(a, b)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        a_wins = (a > b).astype(g.dtype)
        b_wins = (b > a).astype(g.dtype)
        tie = 0.5 * (a == b).astype(g.dtype)
        return g * (a_wins + tie), g * (b_wins + tie)


class Minimum(Function):
    """Elementwise ``min(a, b)``; ties split 0.5/0.5 like :class:`Maximum`."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return np.minimum(a, b)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        a_wins = (a < b).astype(g.dtype)
        b_wins = (b < a).astype(g.dtype)
        tie = 0.5 * (a == b).astype(g.dtype)
        return g * (a_wins + tie), g * (b_wins + tie)


# ---------------------------------------------------------------------------
# Elementwise unary ops
# ---------------------------------------------------------------------------
class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        return -a

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        return -g


class Exp(Function):
    """``y = e^a``; the derivative *is* the output, so we save that, not the input."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = np.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        (out,) = ctx.saved_tensors
        return g * out


class Log(Function):
    """Natural log; dy/da = 1/a."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        (a,) = ctx.saved_tensors
        return g / a


class Abs(Function):
    """``|a|``; derivative ``sign(a)``, with ``sign(0) = 0`` as in PyTorch."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a)
        return np.abs(a)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        (a,) = ctx.saved_tensors
        return g * np.sign(a)


class Clone(Function):
    """Identity with fresh storage; gradient passes straight through."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        return a.copy()

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        return g


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------
class MatMul(Function):
    r"""``y = a @ b``.

    For 2-D operands :math:`Y = AB`, differentiating :math:`L` gives

    .. math:: \bar A = \bar Y B^\top, \qquad \bar B = A^\top \bar Y

    which is just "transpose the *other* operand and contract on the matching
    axis".  Swapping the last two axes generalises this to batched operands;
    any broadcast batch dimensions are then reduced by the engine.

    1-D operands follow NumPy's promotion rules (a leading 1 is prepended to a
    1-D left operand, a trailing 1 appended to a 1-D right operand, and removed
    from the result), so the gradient is reshaped through the same promotion.
    """

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        a2, b2, g2 = a, b, g
        # Re-apply NumPy's 1-D promotion so a single rule covers every case.
        if a.ndim == 1:
            a2 = a[None, :]
            g2 = np.expand_dims(g2, -2 if b.ndim > 1 else -1)
        if b.ndim == 1:
            b2 = b[:, None]
            g2 = np.expand_dims(g2, -1)
        grad_a = g2 @ np.swapaxes(b2, -1, -2)
        grad_b = np.swapaxes(a2, -1, -2) @ g2
        if a.ndim == 1:
            grad_a = grad_a.reshape(grad_a.shape[:-2] + grad_a.shape[-1:])
        if b.ndim == 1:
            grad_b = grad_b.reshape(grad_b.shape[:-1])
        return grad_a, grad_b


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------
class Sum(Function):
    """Sum reduction; the adjoint of a sum is a broadcast of the gradient."""

    @staticmethod
    def forward(
        ctx: Context, a: np.ndarray, *, axis: Any = None, keepdims: bool = False
    ) -> np.ndarray:
        ctx.shape = a.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        return a.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        return _restore_reduced(g, ctx.shape, ctx.axis, ctx.keepdims)


class Mean(Function):
    """Mean reduction: a sum followed by a constant scale, so the gradient is
    the broadcast gradient divided by the number of elements averaged."""

    @staticmethod
    def forward(
        ctx: Context, a: np.ndarray, *, axis: Any = None, keepdims: bool = False
    ) -> np.ndarray:
        ctx.shape = a.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        out = a.mean(axis=axis, keepdims=keepdims)
        ctx.count = a.size // np.asarray(out).size
        return out

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        return _restore_reduced(g / ctx.count, ctx.shape, ctx.axis, ctx.keepdims)


class Max(Function):
    """Max reduction.

    The gradient reaches only the maximal entries.  When several entries tie,
    the gradient is split evenly among them (``torch.amax`` semantics) rather
    than being handed entirely to the first index.
    """

    @staticmethod
    def forward(
        ctx: Context, a: np.ndarray, *, axis: Any = None, keepdims: bool = False
    ) -> np.ndarray:
        out = a.max(axis=axis, keepdims=keepdims)
        ctx.shape = a.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        ctx.save_for_backward(a, np.asarray(out))
        return out

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        a, out = ctx.saved_tensors
        expanded = _restore_reduced(out, ctx.shape, ctx.axis, ctx.keepdims)
        mask = (a == expanded).astype(a.dtype)
        counts = mask.sum(axis=ctx.axis, keepdims=True) if ctx.axis is not None else mask.sum()
        grad = _restore_reduced(g, ctx.shape, ctx.axis, ctx.keepdims)
        return grad * mask / counts


# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------
class Reshape(Function):
    """Pure re-interpretation of storage; the gradient is reshaped straight back."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, *, shape: tuple[int, ...]) -> np.ndarray:
        ctx.shape = a.shape
        return a.reshape(shape)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        return g.reshape(ctx.shape)


class Transpose(Function):
    """Axis permutation; the adjoint is the *inverse* permutation."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, *, axes: tuple[int, ...] | None = None) -> np.ndarray:
        ctx.axes = axes
        return np.transpose(a, axes)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        if ctx.axes is None:
            return np.transpose(g)
        inverse = np.argsort(np.asarray(ctx.axes))
        return np.transpose(g, tuple(inverse))


class GetItem(Function):
    """Indexing / slicing.

    Selecting elements is a linear map that copies some entries and drops the
    rest, so its adjoint scatters the gradient back into a zero array.
    ``np.add.at`` (rather than ``arr[idx] = g``) is essential: with fancy
    indexing the same position can be selected more than once, and those
    contributions must **add**, not overwrite.
    """

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, *, index: Any) -> np.ndarray:
        ctx.shape = a.shape
        ctx.dtype = a.dtype
        ctx.index = index
        return a[index]

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        grad = np.zeros(ctx.shape, dtype=ctx.dtype)
        np.add.at(grad, ctx.index, g)
        return grad


# ---------------------------------------------------------------------------
# Activation primitives
# ---------------------------------------------------------------------------
class ReLU(Function):
    """``max(a, 0)`` with the *PyTorch* subgradient convention at zero.

    ``relu'(0) = 0``.  This differs from ``maximum(a, 0)``, which splits the
    tie 0.5/0.5 -- a real, testable discrepancy that both frameworks share.
    """

    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a)
        return np.maximum(a, 0)

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        (a,) = ctx.saved_tensors
        return g * (a > 0)


class Sigmoid(Function):
    r"""Logistic sigmoid, :math:`\sigma' = \sigma(1-\sigma)`.

    Evaluated branchwise (``exp(x)/(1+exp(x))`` for negative ``x``) so that a
    large negative input underflows to 0 instead of overflowing ``exp(-x)``.
    """

    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        positive = a >= 0
        z = np.exp(-np.abs(a))
        out = np.where(positive, 1.0 / (1.0 + z), z / (1.0 + z)).astype(a.dtype)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        (out,) = ctx.saved_tensors
        return g * out * (1.0 - out)


class Tanh(Function):
    r"""Hyperbolic tangent, :math:`\tanh' = 1 - \tanh^2`."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = np.tanh(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        (out,) = ctx.saved_tensors
        return g * (1.0 - out * out)


# NumPy has no vectorised erf and SciPy is a heavy dependency for one function,
# so we lift the C library's scalar erf. It is exact to ~1 ulp but slow; GELU
# with approximate="tanh" avoids it entirely.
_erf_ufunc = np.frompyfunc(math.erf, 1, 1)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _erf(x: np.ndarray) -> np.ndarray:
    # frompyfunc returns an object-dtype array, so normalise before casting back.
    return np.asarray(_erf_ufunc(x.astype(np.float64)), dtype=np.float64).astype(x.dtype)


class GELU(Function):
    r"""Gaussian Error Linear Unit, :math:`x\,\Phi(x)`.

    Exact form (``approximate="none"``, the PyTorch default)::

        y  = x * 0.5 * (1 + erf(x / sqrt(2)))
        y' = 0.5 * (1 + erf(x / sqrt(2))) + x * phi(x)

    where :math:`\phi` is the standard normal pdf.  The ``"tanh"`` variant is
    Hendrycks & Gimpel's original approximation; its derivative is obtained by
    the product rule on ``0.5 x (1 + tanh u)`` with
    ``u = sqrt(2/pi)(x + 0.044715 x^3)``.
    """

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, *, approximate: str = "none") -> np.ndarray:
        ctx.save_for_backward(a)
        ctx.approximate = approximate
        if approximate == "none":
            return a * 0.5 * (1.0 + _erf(a * _INV_SQRT_2))
        if approximate == "tanh":
            inner = _SQRT_2_OVER_PI * (a + 0.044715 * a**3)
            return 0.5 * a * (1.0 + np.tanh(inner))
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        (a,) = ctx.saved_tensors
        if ctx.approximate == "none":
            cdf = 0.5 * (1.0 + _erf(a * _INV_SQRT_2))
            pdf = _INV_SQRT_2PI * np.exp(-0.5 * a * a)
            return g * (cdf + a * pdf)
        inner = _SQRT_2_OVER_PI * (a + 0.044715 * a**3)
        t = np.tanh(inner)
        d_inner = _SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * a * a)
        return g * (0.5 * (1.0 + t) + 0.5 * a * (1.0 - t * t) * d_inner)


class LogSoftmax(Function):
    r"""Numerically stable :math:`\log \mathrm{softmax}`.

    Computed as :math:`x - m - \log \sum e^{x-m}` with :math:`m = \max x`, so no
    intermediate ever overflows.  The Jacobian collapses to

    .. math:: \bar x = \bar y - e^{y} \textstyle\sum \bar y

    which is why a fused primitive is worth having: the full softmax Jacobian
    is never materialised.
    """

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, *, axis: int = -1) -> np.ndarray:
        shifted = a - a.max(axis=axis, keepdims=True)
        out = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        ctx.save_for_backward(out)
        ctx.axis = axis
        return out

    @staticmethod
    def backward(ctx: Context, g: np.ndarray) -> np.ndarray:
        (out,) = ctx.saved_tensors
        return g - np.exp(out) * g.sum(axis=ctx.axis, keepdims=True)


# ---------------------------------------------------------------------------
# User-facing functional API
# ---------------------------------------------------------------------------
def add(a: Any, b: Any) -> Tensor:
    """Elementwise ``a + b`` with broadcasting."""
    a = _ensure(a, b if isinstance(b, Tensor) else None)
    return Add.apply(a, _ensure(b, a))


def sub(a: Any, b: Any) -> Tensor:
    """Elementwise ``a - b`` with broadcasting."""
    a = _ensure(a, b if isinstance(b, Tensor) else None)
    return Sub.apply(a, _ensure(b, a))


def mul(a: Any, b: Any) -> Tensor:
    """Elementwise ``a * b`` with broadcasting."""
    a = _ensure(a, b if isinstance(b, Tensor) else None)
    return Mul.apply(a, _ensure(b, a))


def div(a: Any, b: Any) -> Tensor:
    """Elementwise ``a / b`` with broadcasting."""
    a = _ensure(a, b if isinstance(b, Tensor) else None)
    return Div.apply(a, _ensure(b, a))


def pow(a: Any, b: Any) -> Tensor:
    """Elementwise ``a ** b``; differentiable in both base and exponent."""
    a = _ensure(a, b if isinstance(b, Tensor) else None)
    return Pow.apply(a, _ensure(b, a))


def neg(a: Tensor) -> Tensor:
    """Elementwise ``-a``."""
    return Neg.apply(_ensure(a))


def matmul(a: Any, b: Any) -> Tensor:
    """Matrix product following NumPy's ``@`` semantics (batched, 1-D promoted)."""
    a = _ensure(a, b if isinstance(b, Tensor) else None)
    return MatMul.apply(a, _ensure(b, a))


def sum(a: Tensor, axis: Any = None, keepdims: bool = False) -> Tensor:
    """Sum over *axis* (all elements if ``None``)."""
    return Sum.apply(_ensure(a), axis=axis, keepdims=keepdims)


def mean(a: Tensor, axis: Any = None, keepdims: bool = False) -> Tensor:
    """Arithmetic mean over *axis*."""
    return Mean.apply(_ensure(a), axis=axis, keepdims=keepdims)


def max(a: Tensor, axis: Any = None, keepdims: bool = False) -> Tensor:
    """Maximum over *axis*; ties share the gradient (``torch.amax`` semantics)."""
    return Max.apply(_ensure(a), axis=axis, keepdims=keepdims)


def reshape(a: Tensor, shape: Any) -> Tensor:
    """Reshape to *shape* (a single ``-1`` is inferred)."""
    if isinstance(shape, int):
        shape = (shape,)
    return Reshape.apply(_ensure(a), shape=tuple(shape))


def transpose(a: Tensor, axes: Any = None) -> Tensor:
    """Permute axes; ``None`` reverses them all."""
    return Transpose.apply(_ensure(a), axes=tuple(axes) if axes is not None else None)


def getitem(a: Tensor, index: Any) -> Tensor:
    """Index or slice, scattering the gradient back on the way down."""
    index = _normalise_index(index)
    return GetItem.apply(_ensure(a), index=index)


def _normalise_index(index: Any) -> Any:
    """Turn any Tensor appearing inside an index into a raw ndarray."""
    if isinstance(index, Tensor):
        return index.data
    if isinstance(index, tuple):
        return tuple(_normalise_index(i) for i in index)
    if isinstance(index, list):
        return [_normalise_index(i) for i in index]
    return index


def exp(a: Tensor) -> Tensor:
    """Elementwise ``e ** a``."""
    return Exp.apply(_ensure(a))


def log(a: Tensor) -> Tensor:
    """Elementwise natural logarithm."""
    return Log.apply(_ensure(a))


def abs(a: Tensor) -> Tensor:
    """Elementwise absolute value."""
    return Abs.apply(_ensure(a))


def clone(a: Tensor) -> Tensor:
    """Differentiable copy."""
    return Clone.apply(_ensure(a))


def maximum(a: Any, b: Any) -> Tensor:
    """Elementwise maximum of two tensors; ties split the gradient evenly."""
    a = _ensure(a, b if isinstance(b, Tensor) else None)
    return Maximum.apply(a, _ensure(b, a))


def minimum(a: Any, b: Any) -> Tensor:
    """Elementwise minimum of two tensors; ties split the gradient evenly."""
    a = _ensure(a, b if isinstance(b, Tensor) else None)
    return Minimum.apply(a, _ensure(b, a))


def relu(a: Tensor) -> Tensor:
    """Rectified linear unit with ``relu'(0) = 0``."""
    return ReLU.apply(_ensure(a))


def sigmoid(a: Tensor) -> Tensor:
    """Logistic sigmoid."""
    return Sigmoid.apply(_ensure(a))


def tanh(a: Tensor) -> Tensor:
    """Hyperbolic tangent."""
    return Tanh.apply(_ensure(a))


def gelu(a: Tensor, approximate: str = "none") -> Tensor:
    """GELU activation; ``approximate`` is ``"none"`` (exact) or ``"tanh"``."""
    return GELU.apply(_ensure(a), approximate=approximate)


def log_softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Stable log-softmax along *axis*."""
    return LogSoftmax.apply(_ensure(a), axis=axis)


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Softmax along *axis*, composed as ``exp(log_softmax(x))`` for stability."""
    return exp(log_softmax(a, axis=axis))
