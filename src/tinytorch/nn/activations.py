"""Activation layers.

Each is a stateless :class:`~tinytorch.nn.module.Module` wrapping the
corresponding primitive in :mod:`tinytorch.ops`, where the derivative lives.
They exist so activations can sit inside a
:class:`~tinytorch.nn.layers.Sequential` alongside layers that do have state.
"""

from __future__ import annotations

from .. import ops
from ..tensor import Tensor
from .module import Module

__all__ = ["GELU", "LeakyReLU", "LogSoftmax", "ReLU", "Sigmoid", "Softmax", "Tanh"]


class ReLU(Module):
    r"""``max(x, 0)``.

    The workhorse: cheap, and its derivative is exactly 0 or 1, so it neither
    shrinks nor inflates gradients through depth -- the reason it displaced
    saturating activations. Convention at the kink is ``relu'(0) = 0``.
    """

    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class LeakyReLU(Module):
    r"""``x`` if ``x > 0`` else ``negative_slope * x``.

    Keeps a small gradient on the negative side so a unit that is pushed
    negative can still recover ("dying ReLU").
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        return ops.maximum(x, x * self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class Sigmoid(Module):
    r""":math:`\sigma(x) = 1/(1+e^{-x})`, mapping to :math:`(0, 1)`.

    Its derivative peaks at 0.25, so stacking sigmoids attenuates gradients by
    at least 4x per layer -- the classic vanishing-gradient problem. Still the
    right choice as the *final* activation for binary classification.
    """

    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)


class Tanh(Module):
    r""":math:`\tanh(x)`, mapping to :math:`(-1, 1)`.

    Zero-centred, with a derivative peaking at 1, so it degrades less than
    sigmoid through depth.
    """

    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class GELU(Module):
    r"""Gaussian Error Linear Unit, :math:`x\,\Phi(x)`.

    A smooth gate: instead of ReLU's hard 0/1 mask, an input is scaled by the
    probability that a standard normal falls below it. Smoothness near the
    origin gives better-behaved gradients, which is why transformers use it.

    ``approximate="none"`` (default) uses the exact ``erf`` form and matches
    PyTorch's default; ``"tanh"`` uses the cheaper original approximation.
    """

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return ops.gelu(x, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


class Softmax(Module):
    """Normalise *axis* into a probability distribution."""

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return ops.softmax(x, axis=self.axis)

    def extra_repr(self) -> str:
        return f"axis={self.axis}"


class LogSoftmax(Module):
    """``log(softmax(x))``, computed stably in one primitive."""

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return ops.log_softmax(x, axis=self.axis)

    def extra_repr(self) -> str:
        return f"axis={self.axis}"
