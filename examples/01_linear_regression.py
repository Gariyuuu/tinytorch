"""Linear regression: fit y = Xw + b and check the answer against algebra.

The point of starting here is that the correct answer is *known*. The data is
generated from a weight vector we hold on to, and the least-squares optimum has
a closed form, so "did it converge" is a checkable claim rather than a
plausible-looking loss curve.

Run:  python examples/01_linear_regression.py
"""

from __future__ import annotations

import numpy as np
from datasets import linear_regression_data

import tinytorch as tt
from tinytorch import nn, optim

EPOCHS = 400
LEARNING_RATE = 0.1
MOMENTUM = 0.9
NOISE = 0.05


def main() -> None:
    tt.manual_seed(0)
    x, y, true_weight, true_bias = linear_regression_data(
        n_samples=400, n_features=3, noise=NOISE, seed=0
    )
    x_t, y_t = tt.Tensor(x), tt.Tensor(y)

    model = nn.Linear(3, 1)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_fn = nn.MSELoss()

    print(f"model: {model}")
    print(f"parameters: {model.num_parameters()}\n")
    print(f"{'epoch':>6}  {'train MSE':>12}")
    print("-" * 22)

    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"{epoch:>6}  {loss.item():>12.6f}")

    # --- Is the answer right? Three independent checks. ---------------------
    fitted_weight = model.weight.data.reshape(-1)
    fitted_bias = float(model.bias.item())

    design = np.hstack([x, np.ones((len(x), 1))])
    closed_form = np.linalg.lstsq(design, y, rcond=None)[0].reshape(-1)
    optimal_loss = float(((design @ closed_form.reshape(-1, 1) - y) ** 2).mean())

    print("\nrecovered parameters vs the values that generated the data")
    print(f"{'':>10}{'true':>12}{'fitted':>12}{'|error|':>12}")
    for i, (t, f) in enumerate(zip(true_weight.reshape(-1), fitted_weight, strict=True)):
        print(f"{'w[' + str(i) + ']':>10}{t:>12.6f}{f:>12.6f}{abs(t - f):>12.2e}")
    print(f"{'b':>10}{true_bias:>12.6f}{fitted_bias:>12.6f}{abs(true_bias - fitted_bias):>12.2e}")

    print("\nvs the least-squares optimum (the best any method could do)")
    print(f"  closed-form MSE : {optimal_loss:.8f}")
    print(f"  achieved MSE    : {loss.item():.8f}")
    print(f"  excess          : {loss.item() - optimal_loss:.2e}")
    print(f"  noise floor     : {NOISE ** 2:.8f}   (irreducible)")


if __name__ == "__main__":
    main()
