r"""Adam: adaptive moment estimation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..autograd import no_grad
from .optimizer import Optimizer

__all__ = ["Adam", "AdamW"]


class Adam(Optimizer):
    r"""Adam (Kingma & Ba, 2015), optionally with AMSGrad.

    Adam keeps two exponential moving averages per parameter -- the mean of the
    gradient and the mean of its square:

    .. math::
        m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2

    and steps by :math:`\eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)`. The
    division makes the *effective* step roughly :math:`\eta` times a unitless
    signal-to-noise ratio, so parameters with small but consistent gradients
    move as fast as parameters with large noisy ones. That per-parameter
    rescaling is why Adam needs so much less learning-rate tuning than SGD.

    **Bias correction** is the subtle part. Both averages start at zero, so
    early on they are biased toward zero: after one step,
    :math:`m_1 = (1-\beta_1)g_1`, which is 10x too small for
    :math:`\beta_1 = 0.9`. Since :math:`\mathbb{E}[m_t] \approx
    (1-\beta_1^t)\mathbb{E}[g]`, dividing by :math:`1-\beta_1^t` removes it
    exactly. The correction decays to 1, which is why it is only visible in the
    first few dozen steps -- and why the step counter ``t`` is genuine
    optimizer state that must be checkpointed.

    ``eps`` sits *outside* the square root here (``sqrt(v_hat) + eps``),
    matching PyTorch. ``amsgrad`` keeps the running maximum of :math:`\hat v`
    so the effective step size is non-increasing, which restores a convergence
    guarantee the original proof was missing.

    Weight decay is coupled L2 (added into the gradient). For decoupled decay
    see :class:`AdamW`.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        beta1, beta2 = betas
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"invalid beta2: {beta2}")
        if eps < 0.0:
            raise ValueError(f"invalid epsilon: {eps}")
        super().__init__(
            params,
            {
                "lr": float(lr),
                "betas": (float(beta1), float(beta2)),
                "eps": float(eps),
                "weight_decay": float(weight_decay),
                "amsgrad": bool(amsgrad),
            },
        )

    #: Set by AdamW to use decoupled weight decay.
    decoupled_weight_decay = False

    @no_grad()
    def step(self) -> None:
        """Apply one Adam update in place."""
        for group, param in self._params():
            if param.grad is None:
                continue
            dtype = param.data.dtype
            grad = param.grad.astype(dtype, copy=False)
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            if weight_decay != 0.0:
                if self.decoupled_weight_decay:
                    # AdamW: shrink the weight directly, untouched by the
                    # adaptive denominator.
                    param.data -= lr * weight_decay * param.data
                else:
                    grad = grad + weight_decay * param.data

            state = self._get_state(param)
            if not state:
                state["step"] = 0
                state["exp_avg"] = np.zeros_like(param.data)
                state["exp_avg_sq"] = np.zeros_like(param.data)
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = np.zeros_like(param.data)

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            exp_avg *= beta1
            exp_avg += (1.0 - beta1) * grad
            exp_avg_sq *= beta2
            exp_avg_sq += (1.0 - beta2) * (grad * grad)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step

            if group["amsgrad"]:
                np.maximum(state["max_exp_avg_sq"], exp_avg_sq, out=state["max_exp_avg_sq"])
                second_moment = state["max_exp_avg_sq"]
            else:
                second_moment = exp_avg_sq

            denom = np.sqrt(second_moment) / np.sqrt(bias_correction2) + group["eps"]
            param.data -= (lr / bias_correction1) * (exp_avg / denom)


class AdamW(Adam):
    r"""Adam with **decoupled** weight decay (Loshchilov & Hutter, 2019).

    Classic Adam folds L2 into the gradient, where it is then divided by
    :math:`\sqrt{\hat v}` -- so parameters with large gradient variance get
    *less* regularisation, which is the opposite of the intent. AdamW instead
    applies :math:`\theta \leftarrow \theta(1 - \eta\lambda)` directly, keeping
    decay independent of the adaptive scaling.
    """

    decoupled_weight_decay = True

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
