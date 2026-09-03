r"""Stochastic gradient descent, with optional momentum and Nesterov lookahead."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..autograd import no_grad
from .optimizer import Optimizer

__all__ = ["SGD"]


class SGD(Optimizer):
    r"""SGD with momentum, dampening, weight decay and Nesterov acceleration.

    Plain SGD takes :math:`\theta \leftarrow \theta - \eta g`. Its weakness is
    anisotropy: in a ravine (curvature much larger across the valley than along
    it) the step size is capped by the steep direction, so progress along the
    shallow direction crawls.

    **Momentum** fixes that by accumulating an exponentially weighted velocity,

    .. math:: v \leftarrow \mu v + (1-\tau) g, \qquad \theta \leftarrow \theta - \eta v

    Oscillating components cancel between steps while the consistent component
    compounds toward :math:`g/(1-\mu)` -- an effective step up to
    :math:`1/(1-\mu)` times larger along the direction that keeps agreeing.

    **Nesterov** evaluates the gradient after the momentum step rather than
    before, which in this bookkeeping becomes :math:`g \leftarrow g + \mu v`:
    a correction term that damps overshoot as the velocity turns.

    The update is written to exactly match ``torch.optim.SGD``, including two
    details that are easy to get wrong:

    * the momentum buffer is **initialised to the gradient itself** on the
      first step (not to zeros, which would halve the first step), and
    * dampening is not applied on that first step.

    Weight decay is the classic L2 form ``g += weight_decay * theta``, folded
    into the gradient *before* momentum -- so it, too, is smoothed by the
    velocity. (This is not decoupled/AdamW-style decay.)

    Parameters
    ----------
    params: iterable of parameters or parameter-group dicts.
    lr: learning rate :math:`\eta`. Required; there is no safe default.
    momentum: :math:`\mu`, typically 0.9. ``0`` disables the velocity buffer.
    dampening: :math:`\tau`, damps the incoming gradient's contribution.
    weight_decay: L2 penalty coefficient.
    nesterov: use the lookahead correction (requires momentum > 0, dampening 0).
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("nesterov momentum requires momentum > 0 and dampening == 0")
        super().__init__(
            params,
            {
                "lr": float(lr),
                "momentum": float(momentum),
                "dampening": float(dampening),
                "weight_decay": float(weight_decay),
                "nesterov": bool(nesterov),
            },
        )

    @no_grad()
    def step(self) -> None:
        """Apply one SGD update in place."""
        for group, param in self._params():
            if param.grad is None:
                continue
            grad = param.grad.astype(param.data.dtype, copy=False)

            if group["weight_decay"] != 0.0:
                grad = grad + group["weight_decay"] * param.data

            momentum = group["momentum"]
            if momentum != 0.0:
                state = self._get_state(param)
                buffer = state.get("momentum_buffer")
                if buffer is None:
                    # First step: seed with the gradient, matching PyTorch.
                    buffer = np.array(grad, copy=True)
                else:
                    buffer = momentum * buffer + (1.0 - group["dampening"]) * grad
                state["momentum_buffer"] = buffer
                grad = grad + momentum * buffer if group["nesterov"] else buffer

            param.data -= group["lr"] * grad
