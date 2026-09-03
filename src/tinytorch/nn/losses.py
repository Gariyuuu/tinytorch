"""Loss functions.

A loss is just a :class:`~tinytorch.nn.module.Module` returning a scalar -- the
root of the backward pass.  Nothing special happens here; the losses are built
from the same primitives as everything else, which is exactly the point: the
autograd engine is what makes them differentiable, not a hand-written
``backward``.

The ``reduction`` argument decides how per-sample losses become one number:
``"mean"`` (default) keeps the gradient scale independent of batch size,
``"sum"`` does not, and ``"none"`` returns the per-sample vector.
"""

from __future__ import annotations

import numpy as np

from .. import ops
from ..tensor import Tensor
from .module import Module

__all__ = ["BCELoss", "CrossEntropyLoss", "L1Loss", "MSELoss", "NLLLoss"]

_REDUCTIONS = ("mean", "sum", "none")


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return ops.mean(loss)
    if reduction == "sum":
        return ops.sum(loss)
    if reduction == "none":
        return loss
    raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")


class _Loss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
        self.reduction = reduction

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class MSELoss(_Loss):
    r"""Mean squared error, :math:`\frac{1}{N}\sum (\hat y - y)^2`.

    The gradient w.r.t. the prediction is :math:`2(\hat y - y)/N` -- linear in
    the error, so a prediction twice as wrong pulls twice as hard. That
    linearity is what makes MSE the natural loss for regression (it is the
    negative log-likelihood of Gaussian noise) and also what makes it sensitive
    to outliers.
    """

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = ops.sub(prediction, target)
        return _reduce(ops.mul(diff, diff), self.reduction)


class L1Loss(_Loss):
    r"""Mean absolute error. Constant-magnitude gradient, so outliers do not
    dominate the way they do under :class:`MSELoss`."""

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return _reduce(ops.abs(ops.sub(prediction, target)), self.reduction)


class NLLLoss(_Loss):
    r"""Negative log-likelihood over **log-probabilities**.

    Given ``log_probs`` of shape ``(N, C)`` and integer ``target`` of shape
    ``(N,)``, returns :math:`-\frac{1}{N}\sum_n \text{log\_probs}[n, y_n]`.

    The gather is an indexing op, so its gradient is a scatter -- exactly one
    entry per row receives gradient. Combine with
    :class:`~tinytorch.nn.activations.LogSoftmax` to get
    :class:`CrossEntropyLoss`.
    """

    def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
        indices = target.data if isinstance(target, Tensor) else np.asarray(target)
        indices = indices.astype(np.intp).reshape(-1)
        if log_probs.ndim != 2:
            raise ValueError(f"NLLLoss expects (N, C) log-probabilities, got {log_probs.shape}")
        rows = np.arange(log_probs.shape[0], dtype=np.intp)
        picked = ops.getitem(log_probs, (rows, indices))
        return _reduce(ops.neg(picked), self.reduction)


class CrossEntropyLoss(_Loss):
    r"""Softmax cross-entropy over raw **logits**.

    Composed as ``NLLLoss(log_softmax(logits), target)``, which is the
    numerically sound way to do it: computing ``log(softmax(x))`` as two
    separate steps overflows for large logits and produces ``-inf`` for tiny
    probabilities, whereas the fused
    :class:`~tinytorch.ops.LogSoftmax` subtracts the row max first.

    Composing rather than fusing the whole loss also demonstrates that the
    well-known result

    .. math::

        \frac{\partial L}{\partial \text{logits}}
            = \frac{\text{softmax}(z) - \text{onehot}(y)}{N}

    is *derived* by the engine from log-softmax's and gather's local rules --
    it is not special-cased anywhere.
    """

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError(f"CrossEntropyLoss expects (N, C) logits, got {logits.shape}")
        log_probs = ops.log_softmax(logits, axis=-1)
        return NLLLoss(self.reduction)(log_probs, target)


class BCELoss(_Loss):
    r"""Binary cross-entropy over **probabilities** in :math:`(0, 1)`.

    .. math:: L = -[y \log p + (1-y)\log(1-p)]

    Probabilities are clamped away from the boundary before the log, because
    ``log(0)`` is ``-inf`` and would poison the whole batch's gradient. If you
    have logits, prefer a numerically stable logits-based formulation.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-12) -> None:
        super().__init__(reduction)
        self.eps = float(eps)

    def forward(self, probability: Tensor, target: Tensor) -> Tensor:
        p = ops.minimum(ops.maximum(probability, self.eps), 1.0 - self.eps)
        loss = ops.neg(
            ops.add(
                ops.mul(target, ops.log(p)),
                ops.mul(ops.sub(1.0, target), ops.log(ops.sub(1.0, p))),
            )
        )
        return _reduce(loss, self.reduction)
