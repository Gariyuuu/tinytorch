"""Extend the framework: define a new differentiable primitive.

Anything that subclasses ``tinytorch.Function`` and implements the two static
maps participates in the graph exactly like a built-in op -- including
composition, broadcasting and gradient accumulation. This example adds ``sin``,
verifies it numerically, and trains with it.

Run:  python examples/05_custom_function.py
"""

from __future__ import annotations

import numpy as np

import tinytorch as tt
from tinytorch import nn, optim


class Sin(tt.Function):
    """y = sin(x); dy/dx = cos(x)."""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.sin(a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output * np.cos(a)


def sin(x: tt.Tensor) -> tt.Tensor:
    return Sin.apply(x)


class SinLayer(nn.Module):
    """A module wrapping the new primitive, usable inside Sequential."""

    def forward(self, x):
        return sin(x)


def main() -> None:
    tt.set_default_dtype(np.float64)

    # 1. The new op composes with existing ones and differentiates correctly.
    x = tt.Tensor([0.0, np.pi / 6, np.pi / 2], requires_grad=True)
    (sin(x) * 2.0).sum().backward()
    print("d/dx [2 sin x] at [0, pi/6, pi/2]")
    print(f"  analytic : {x.grad}")
    print(f"  expected : {2 * np.cos(x.data)}")

    # 2. Verify it against finite differences -- the same check the built-in
    #    ops are held to.
    ok = tt.gradcheck(lambda t: sin(t) * t, [tt.randn(4, 5, requires_grad=True)])
    print(f"\ngradcheck(sin(x) * x): {ok}")

    # 3. Use it as a real activation and fit a nonlinear target.
    tt.manual_seed(31)
    inputs = np.linspace(-3.0, 3.0, 256).reshape(-1, 1)
    targets = np.sin(3.0 * inputs) * np.exp(-0.2 * inputs**2)

    model = nn.Sequential(
        nn.Linear(1, 32), SinLayer(), nn.Linear(32, 32), SinLayer(), nn.Linear(32, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    x_t, y_t = tt.Tensor(inputs), tt.Tensor(targets)

    print(
        "\nfitting sin(3x)exp(-0.2x^2) with a sin-activated MLP "
        f"({model.num_parameters()} params)"
    )
    print(f"{'epoch':>6}  {'MSE':>12}")
    print("-" * 22)
    for epoch in range(2001):
        optimizer.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        optimizer.step()
        if epoch % 400 == 0:
            print(f"{epoch:>6}  {loss.item():>12.8f}")

    variance = float(targets.var())
    print(f"\ntarget variance (loss of predicting the mean): {variance:.6f}")
    print(f"final MSE                                    : {loss.item():.8f}")
    print(f"variance explained                           : {1 - loss.item() / variance:.4%}")


if __name__ == "__main__":
    main()
