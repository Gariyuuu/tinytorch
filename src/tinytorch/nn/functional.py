"""Stateless functional forms of the nn layers (``nn.functional`` by habit)."""

from __future__ import annotations

from typing import Any

from .. import ops
from ..tensor import Tensor
from .losses import CrossEntropyLoss, MSELoss, NLLLoss

__all__ = [
    "cross_entropy",
    "gelu",
    "linear",
    "log_softmax",
    "mse_loss",
    "nll_loss",
    "relu",
    "sigmoid",
    "softmax",
    "tanh",
]

relu = ops.relu
sigmoid = ops.sigmoid
tanh = ops.tanh
gelu = ops.gelu
softmax = ops.softmax
log_softmax = ops.log_softmax


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """``x @ weight.T + bias`` -- the functional core of ``nn.Linear``."""
    out = ops.matmul(x, ops.transpose(weight))
    return out if bias is None else ops.add(out, bias)


def mse_loss(prediction: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Functional :class:`~tinytorch.nn.losses.MSELoss`."""
    return MSELoss(reduction)(prediction, target)


def nll_loss(log_probs: Tensor, target: Any, reduction: str = "mean") -> Tensor:
    """Functional :class:`~tinytorch.nn.losses.NLLLoss`."""
    return NLLLoss(reduction)(log_probs, target)


def cross_entropy(logits: Tensor, target: Any, reduction: str = "mean") -> Tensor:
    """Functional :class:`~tinytorch.nn.losses.CrossEntropyLoss`."""
    return CrossEntropyLoss(reduction)(logits, target)
