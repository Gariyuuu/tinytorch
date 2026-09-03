"""XOR: the smallest problem that genuinely needs a hidden layer.

XOR is not linearly separable, so a single ``Linear`` layer is provably stuck
near 50%. This example trains both a linear model and a one-hidden-layer MLP on
the same data, so the difference is attributable to the nonlinearity and to
backprop flowing through it -- not to luck.

Run:  python examples/02_xor_classification.py
"""

from __future__ import annotations

import numpy as np
from datasets import xor_data

import tinytorch as tt
from tinytorch import nn, optim

EPOCHS = 600
LEARNING_RATE = 0.05


def accuracy(model: nn.Module, x: tt.Tensor, y: np.ndarray) -> float:
    model.eval()
    with tt.no_grad():
        predictions = model(x).data.argmax(axis=1)
    model.train()
    return float((predictions == y).mean())


def train(model: nn.Module, x: tt.Tensor, y_t: tt.Tensor, y: np.ndarray, label: str) -> float:
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    print(f"\n{label}  ({model.num_parameters()} parameters)")
    print(f"{'epoch':>6}  {'loss':>10}  {'accuracy':>9}")
    print("-" * 30)
    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y_t)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"{epoch:>6}  {loss.item():>10.5f}  {accuracy(model, x, y):>9.3f}")
    return accuracy(model, x, y)


def decision_grid(model: nn.Module, resolution: int = 21) -> str:
    """ASCII picture of the learned decision boundary."""
    axis = np.linspace(-2.0, 2.0, resolution)
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    with tt.no_grad():
        predictions = model(tt.Tensor(grid)).data.argmax(axis=1).reshape(resolution, resolution)
    rows = ["".join("#" if v else "." for v in row) for row in predictions[::-1]]
    return "\n".join("  " + row for row in rows)


def main() -> None:
    tt.manual_seed(2)
    x, y = xor_data(n_per_quadrant=100, spread=0.25, seed=1)
    x_t, y_t = tt.Tensor(x.astype(np.float64)), tt.Tensor(y)

    linear = nn.Linear(2, 2)
    linear_accuracy = train(linear, x_t, y_t, y, "linear model (no hidden layer)")

    tt.manual_seed(2)
    mlp = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
    mlp_accuracy = train(mlp, x_t, y_t, y, "MLP (one hidden layer, tanh)")

    print("\n" + "=" * 46)
    print(f"linear model accuracy : {linear_accuracy:.3f}  (chance is 0.500)")
    print(f"MLP accuracy          : {mlp_accuracy:.3f}")
    print("=" * 46)
    print("\nMLP decision boundary over [-2, 2]^2 ('#' = class 1):")
    print(decision_grid(mlp))


if __name__ == "__main__":
    main()
