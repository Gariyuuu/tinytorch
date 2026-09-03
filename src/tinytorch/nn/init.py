r"""Parameter initialisation schemes.

Why initialisation is not an afterthought
-----------------------------------------
A deep stack of linear maps multiplies variances.  If each layer scales the
signal by :math:`k`, then after :math:`L` layers the signal is scaled by
:math:`k^L` -- exponential blow-up for :math:`k>1`, exponential decay for
:math:`k<1`.  The fix is to choose the weight variance so that :math:`k \approx 1`.

**Xavier/Glorot** (Glorot & Bengio, 2010) targets unit variance in *both*
directions at once and so compromises between them:

.. math:: \operatorname{Var}(W) = \frac{2}{\text{fan\_in} + \text{fan\_out}}

It assumes the activation is roughly linear near the origin, which holds for
``tanh`` and ``sigmoid`` but not for ``ReLU``.

**Kaiming/He** (He et al., 2015) accounts for ReLU zeroing half the inputs,
which halves the variance, and compensates:

.. math:: \operatorname{Var}(W) = \frac{2}{\text{fan\_in}}

The ``gain`` in :func:`calculate_gain` is the per-nonlinearity correction
factor; ``sqrt(2)`` for ReLU is exactly that "half the units are dead" term.

Fan computation
---------------
``fan_in`` is the number of inputs feeding one output unit and ``fan_out`` the
number of outputs a single input feeds.  For a weight of shape
``(fan_out, fan_in)`` -- TinyTorch's :class:`~tinytorch.nn.layers.Linear`
layout, matching PyTorch -- that reads straight off the two dimensions.

Every function here mutates its argument in place and returns it, and every
name ends in ``_`` to say so.
"""

from __future__ import annotations

import math

import numpy as np

from ..creation import default_generator
from ..tensor import Tensor

__all__ = [
    "calculate_fan_in_and_fan_out",
    "calculate_gain",
    "constant_",
    "kaiming_normal_",
    "kaiming_uniform_",
    "normal_",
    "ones_",
    "uniform_",
    "xavier_normal_",
    "xavier_uniform_",
    "zeros_",
]


def calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    """Return ``(fan_in, fan_out)`` for a weight tensor.

    Dimensions beyond the first two (a convolution's kernel window, say) are
    part of the *receptive field* and multiply both fans.
    """
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("fan cannot be computed for a tensor with fewer than 2 dimensions")
    fan_out, fan_in = shape[0], shape[1]
    receptive_field = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    return fan_in * receptive_field, fan_out * receptive_field


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    """Recommended variance-preserving gain for a nonlinearity.

    Values follow ``torch.nn.init.calculate_gain``.
    """
    linear = {"linear", "conv1d", "conv2d", "conv3d", "identity", "sigmoid"}
    if nonlinearity in linear:
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1.0 + slope**2))
    if nonlinearity == "selu":
        return 3.0 / 4.0
    raise ValueError(f"unsupported nonlinearity {nonlinearity!r}")


def _fill(tensor: Tensor, values: np.ndarray) -> Tensor:
    tensor.data[...] = values.astype(tensor.data.dtype, copy=False)
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    """Fill with 0. Correct for biases, catastrophic for weights (all units
    would compute the same thing and receive the same gradient forever)."""
    tensor.data[...] = 0
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    """Fill with 1 (normalisation scales)."""
    tensor.data[...] = 1
    return tensor


def constant_(tensor: Tensor, value: float) -> Tensor:
    """Fill with a constant."""
    tensor.data[...] = value
    return tensor


def uniform_(tensor: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    """Fill from :math:`\\mathcal{U}(low, high)`."""
    return _fill(tensor, default_generator().uniform(low, high, size=tensor.shape))


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill from :math:`\\mathcal{N}(mean, std^2)`."""
    return _fill(tensor, default_generator().normal(mean, std, size=tensor.shape))


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Glorot uniform: bound ``gain * sqrt(6 / (fan_in + fan_out))``.

    The 6 comes from matching variances: :math:`\\mathcal{U}(-a, a)` has
    variance :math:`a^2/3`, so :math:`a = \\sqrt{3\\,\\mathrm{Var}}` and
    :math:`\\mathrm{Var} = 2/(\\text{fan_in}+\\text{fan_out})` gives
    :math:`a = \\sqrt{6/(\\ldots)}`.
    """
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Glorot normal: ``std = gain * sqrt(2 / (fan_in + fan_out))``."""
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    return normal_(tensor, 0.0, gain * math.sqrt(2.0 / (fan_in + fan_out)))


def _resolve_fan(tensor: Tensor, mode: str) -> int:
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        return fan_in
    if mode == "fan_out":
        return fan_out
    raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    """He uniform: bound ``sqrt(3) * gain / sqrt(fan)``.

    ``mode="fan_in"`` preserves variance on the forward pass; ``"fan_out"``
    preserves it on the backward pass.
    """
    fan = _resolve_fan(tensor, mode)
    bound = math.sqrt(3.0) * calculate_gain(nonlinearity, a) / math.sqrt(fan)
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    """He normal: ``std = gain / sqrt(fan)``."""
    fan = _resolve_fan(tensor, mode)
    return normal_(tensor, 0.0, calculate_gain(nonlinearity, a) / math.sqrt(fan))
