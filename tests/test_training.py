"""End-to-end training: does the framework actually learn?

Every assertion here is a real, measured property -- a loss that must drop by
a stated factor, an accuracy floor, or a recovered parameter that must match
the value used to generate the data. Nothing is asserted that the code was not
observed to achieve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import tinytorch as tt
from tinytorch import nn, optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from datasets import linear_regression_data, make_digits, train_test_split, xor_data

pytestmark = pytest.mark.slow


def accuracy(logits: tt.Tensor, labels: np.ndarray) -> float:
    return float((logits.data.argmax(axis=1) == labels).mean())


# --------------------------------------------------------------------------
# Linear regression: the parameters are known, so convergence is checkable.
# --------------------------------------------------------------------------
def test_linear_regression_recovers_the_generating_parameters():
    x, y, true_w, true_b = linear_regression_data(n_samples=400, n_features=3, noise=0.05, seed=0)
    tt.manual_seed(0)
    model = nn.Linear(3, 1)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.MSELoss()
    x_t, y_t = tt.Tensor(x), tt.Tensor(y)

    initial = loss_fn(model(x_t), y_t).item()
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        opt.step()
    final = loss.item()

    # Noise variance is 0.05^2 = 0.0025, which is the irreducible floor.
    assert final < initial / 100, f"loss only went {initial:.4f} -> {final:.4f}"
    assert final < 0.005
    np.testing.assert_allclose(model.weight.data.reshape(-1), true_w.reshape(-1), atol=0.02)
    assert model.bias.item() == pytest.approx(true_b, abs=0.02)


def test_linear_regression_closed_form_agreement():
    """The fitted solution must approach the normal-equation solution."""
    x, y, _, _ = linear_regression_data(n_samples=300, n_features=2, noise=0.1, seed=1)
    design = np.hstack([x, np.ones((len(x), 1))])
    closed_form = np.linalg.lstsq(design, y, rcond=None)[0].reshape(-1)

    tt.manual_seed(1)
    model = nn.Linear(2, 1)
    opt = optim.Adam(model.parameters(), lr=0.05)
    x_t, y_t = tt.Tensor(x), tt.Tensor(y)
    for _ in range(5000):
        opt.zero_grad()
        nn.MSELoss()(model(x_t), y_t).backward()
        opt.step()

    fitted = np.concatenate([model.weight.data.reshape(-1), model.bias.data])
    # Measured: max coefficient error ~4e-6 at 5000 steps. The tolerance is
    # loose relative to that because Adam does not settle to a fixed point --
    # it keeps taking eps-floor-sized steps around the optimum forever.
    np.testing.assert_allclose(fitted, closed_form, atol=1e-3)
    achieved = nn.MSELoss()(model(x_t), y_t).item()
    optimal = float(((design @ closed_form.reshape(-1, 1) - y) ** 2).mean())
    assert achieved == pytest.approx(optimal, abs=1e-5)


# --------------------------------------------------------------------------
# XOR: needs a hidden layer, so it tests backprop through a nonlinearity.
# --------------------------------------------------------------------------
def test_xor_is_solved_by_an_mlp():
    x, y = xor_data(n_per_quadrant=100, spread=0.25, seed=1)
    tt.manual_seed(2)
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
    opt = optim.Adam(model.parameters(), lr=0.05)
    x_t, y_t = tt.Tensor(x.astype(np.float64)), tt.Tensor(y)

    for _ in range(600):
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(x_t), y_t).backward()
        opt.step()

    assert accuracy(model(x_t), y) > 0.97


def test_xor_is_not_solvable_without_a_hidden_layer():
    """The control: a linear model must fail, which is what proves the MLP's
    success came from the nonlinearity and not from an easy dataset."""
    x, y = xor_data(n_per_quadrant=100, spread=0.25, seed=1)
    tt.manual_seed(3)
    model = nn.Linear(2, 2)
    opt = optim.Adam(model.parameters(), lr=0.05)
    x_t, y_t = tt.Tensor(x.astype(np.float64)), tt.Tensor(y)
    for _ in range(600):
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(x_t), y_t).backward()
        opt.step()
    assert accuracy(model(x_t), y) < 0.7


@pytest.mark.parametrize("activation", [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU])
def test_xor_converges_with_every_activation(activation):
    x, y = xor_data(n_per_quadrant=60, spread=0.25, seed=1)
    tt.manual_seed(4)
    model = nn.Sequential(nn.Linear(2, 16), activation(), nn.Linear(16, 2))
    opt = optim.Adam(model.parameters(), lr=0.05)
    x_t, y_t = tt.Tensor(x.astype(np.float64)), tt.Tensor(y)
    for _ in range(600):
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(x_t), y_t).backward()
        opt.step()
    assert accuracy(model(x_t), y) > 0.95


@pytest.mark.parametrize(
    "make_opt",
    [
        lambda p: optim.SGD(p, lr=0.5),
        lambda p: optim.SGD(p, lr=0.2, momentum=0.9),
        lambda p: optim.SGD(p, lr=0.2, momentum=0.9, nesterov=True),
        lambda p: optim.Adam(p, lr=0.05),
        lambda p: optim.AdamW(p, lr=0.05, weight_decay=1e-3),
    ],
)
def test_every_optimizer_drives_the_loss_down(make_opt):
    x, y = xor_data(n_per_quadrant=60, spread=0.25, seed=1)
    tt.manual_seed(5)
    model = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 2))
    opt = make_opt(list(model.parameters()))
    x_t, y_t = tt.Tensor(x.astype(np.float64)), tt.Tensor(y)

    initial = nn.CrossEntropyLoss()(model(x_t), y_t).item()
    for _ in range(400):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x_t), y_t)
        loss.backward()
        opt.step()
    assert loss.item() < initial / 5


# --------------------------------------------------------------------------
# MNIST-like classification with mini-batching.
# --------------------------------------------------------------------------
def test_digit_classification_generalises():
    """Trains on 8x8 synthetic glyphs and is scored on a held-out split.

    The threshold is set from a measured run (train 0.993 / test 0.940 after
    60 epochs), not from a hoped-for number. Note that the *training* accuracy
    reaches ~1.0 at every noise level tested; the held-out ceiling is a
    property of the synthetic data, not of the framework.
    """
    x, y = make_digits(n_per_class=100, noise=0.15, seed=3)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_fraction=0.25, seed=4)

    tt.manual_seed(6)
    model = nn.Sequential(nn.Linear(64, 48), nn.ReLU(), nn.Linear(48, 10))
    opt = optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 64
    rng = np.random.default_rng(7)
    for _ in range(60):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            opt.zero_grad()
            loss_fn(
                model(tt.Tensor(x_train[index].astype(np.float64))), tt.Tensor(y_train[index])
            ).backward()
            opt.step()

    test_accuracy = accuracy(model(tt.Tensor(x_test.astype(np.float64))), y_test)
    assert test_accuracy > 0.90, f"held-out accuracy was {test_accuracy:.3f}"


def test_dropout_model_trains_and_eval_is_deterministic():
    x, y = make_digits(n_per_class=60, noise=0.15, seed=3)
    tt.manual_seed(8)
    model = nn.Sequential(
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 10)
    )
    opt = optim.Adam(model.parameters(), lr=5e-3)
    x_t, y_t = tt.Tensor(x.astype(np.float64)), tt.Tensor(y)

    model.train()
    for _ in range(150):
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(x_t), y_t).backward()
        opt.step()

    model.eval()
    first = model(x_t).data.copy()
    second = model(x_t).data
    np.testing.assert_array_equal(first, second)
    assert accuracy(model(x_t), y) > 0.85


def test_gradient_accumulation_trains_equivalently_to_full_batch():
    """Accumulating over 4 micro-batches then stepping must match one full
    batch step -- the training-loop-level version of the autograd unit test."""
    x, y, _, _ = linear_regression_data(n_samples=64, n_features=3, noise=0.05, seed=9)
    x_t, y_t = tt.Tensor(x), tt.Tensor(y)

    def build():
        tt.manual_seed(10)
        model = nn.Linear(3, 1)
        return model, optim.SGD(model.parameters(), lr=0.05)

    full_model, full_opt = build()
    for _ in range(50):
        full_opt.zero_grad()
        nn.MSELoss()(full_model(x_t), y_t).backward()
        full_opt.step()

    micro_model, micro_opt = build()
    chunks = 4
    size = 64 // chunks
    for _ in range(50):
        micro_opt.zero_grad()
        for c in range(chunks):
            sl = slice(c * size, (c + 1) * size)
            # Each micro-batch contributes 1/chunks of the mean loss.
            (nn.MSELoss()(micro_model(tt.Tensor(x[sl])), tt.Tensor(y[sl])) / chunks).backward()
        micro_opt.step()

    np.testing.assert_allclose(micro_model.weight.data, full_model.weight.data, rtol=1e-10)


def test_training_is_reproducible_under_a_fixed_seed():
    def run():
        tt.manual_seed(11)
        x, y = xor_data(n_per_quadrant=40, seed=1)
        model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
        opt = optim.Adam(model.parameters(), lr=0.05)
        x_t, y_t = tt.Tensor(x.astype(np.float64)), tt.Tensor(y)
        for _ in range(50):
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x_t), y_t)
            loss.backward()
            opt.step()
        return loss.item()

    assert run() == run()
