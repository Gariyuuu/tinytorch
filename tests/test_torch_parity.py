"""TinyTorch vs PyTorch: forward values, gradients and optimizer updates.

Gradient checking proves TinyTorch is self-consistent. This file proves it
computes the *same function* as a reference implementation -- including
conventions that finite differences cannot see (the subgradient at a kink, tie
handling, PyTorch's exact momentum bookkeeping, bias-correction order).

Method
------
Every test builds the identical computation twice, seeds both from the *same*
NumPy array so the inputs are bit-identical, runs both backward passes, and
compares in float64 at ``rtol=1e-10`` (see ``conftest`` for the tolerance
rationale). PyTorch is a test-only dependency; the whole module skips if it is
not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytorch as tt
from tinytorch import nn, optim

from .conftest import TORCH_F32_ATOL, TORCH_F32_RTOL, TORCH_F64_ATOL, TORCH_F64_RTOL

torch = pytest.importorskip("torch", reason="PyTorch parity tests require torch")
pytestmark = pytest.mark.torch


# --------------------------------------------------------------------------
# Helpers: build the same tensor in both frameworks.
# --------------------------------------------------------------------------
def pair(array, requires_grad: bool = True):
    """Return ``(tinytorch_tensor, torch_tensor)`` sharing identical values."""
    array = np.asarray(array, dtype=np.float64)
    mine = tt.Tensor(array.copy(), requires_grad=requires_grad)
    theirs = torch.tensor(array.copy(), dtype=torch.float64, requires_grad=requires_grad)
    return mine, theirs


def randn(*shape, seed: int = 0):
    return np.random.default_rng(seed).standard_normal(shape)


def close(mine, theirs, rtol=TORCH_F64_RTOL, atol=TORCH_F64_ATOL, what="value"):
    mine_np = mine.data if isinstance(mine, tt.Tensor) else np.asarray(mine)
    theirs_np = theirs.detach().cpu().numpy() if torch.is_tensor(theirs) else np.asarray(theirs)
    np.testing.assert_allclose(
        np.asarray(mine_np, dtype=np.float64),
        np.asarray(theirs_np, dtype=np.float64),
        rtol=rtol,
        atol=atol,
        err_msg=f"{what} disagrees with PyTorch",
    )


def compare_unary(fn_tt, fn_torch, array, rtol=TORCH_F64_RTOL, atol=TORCH_F64_ATOL):
    """Run a one-input op in both frameworks; compare forward and gradient."""
    mine, theirs = pair(array)
    out_mine, out_theirs = fn_tt(mine), fn_torch(theirs)
    close(out_mine, out_theirs, rtol, atol, "forward")
    out_mine.sum().backward()
    out_theirs.sum().backward()
    close(mine.grad, theirs.grad, rtol, atol, "gradient")


def compare_binary(fn_tt, fn_torch, a_arr, b_arr, rtol=TORCH_F64_RTOL, atol=TORCH_F64_ATOL):
    """Run a two-input op in both frameworks; compare forward and both grads."""
    a_mine, a_theirs = pair(a_arr)
    b_mine, b_theirs = pair(b_arr)
    out_mine, out_theirs = fn_tt(a_mine, b_mine), fn_torch(a_theirs, b_theirs)
    close(out_mine, out_theirs, rtol, atol, "forward")
    out_mine.sum().backward()
    out_theirs.sum().backward()
    close(a_mine.grad, a_theirs.grad, rtol, atol, "grad wrt a")
    close(b_mine.grad, b_theirs.grad, rtol, atol, "grad wrt b")


# --------------------------------------------------------------------------
# Elementwise ops.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "mine", "theirs", "offset"),
    [
        ("add", lambda a, b: a + b, lambda a, b: a + b, 0.0),
        ("sub", lambda a, b: a - b, lambda a, b: a - b, 0.0),
        ("mul", lambda a, b: a * b, lambda a, b: a * b, 0.0),
        ("div", lambda a, b: a / b, lambda a, b: a / b, 4.0),
        ("pow", lambda a, b: a**b, lambda a, b: a**b, 4.0),
        ("maximum", tt.maximum, torch.maximum, 0.0),
        ("minimum", tt.minimum, torch.minimum, 0.0),
        ("matmul_elemwise_chain", lambda a, b: (a * b) / (a + b + 10.0),
         lambda a, b: (a * b) / (a + b + 10.0), 0.0),
    ],
)
def test_binary_op_parity(name, mine, theirs, offset):
    compare_binary(mine, theirs, randn(4, 3, seed=1) + offset, randn(4, 3, seed=2) + offset)


@pytest.mark.parametrize(
    ("name", "mine", "theirs", "offset"),
    [
        ("neg", lambda t: -t, lambda t: -t, 0.0),
        ("exp", tt.exp, torch.exp, 0.0),
        ("log", tt.log, torch.log, 5.0),
        ("abs", tt.ops.abs, torch.abs, 0.0),
        ("sqrt", lambda t: t.sqrt(), torch.sqrt, 5.0),
        ("relu", tt.relu, torch.relu, 0.0),
        ("sigmoid", tt.sigmoid, torch.sigmoid, 0.0),
        ("tanh", tt.tanh, torch.tanh, 0.0),
        ("gelu_exact", tt.gelu, torch.nn.functional.gelu, 0.0),
        ("gelu_tanh", lambda t: tt.gelu(t, approximate="tanh"),
         lambda t: torch.nn.functional.gelu(t, approximate="tanh"), 0.0),
        ("softmax", lambda t: tt.softmax(t, -1),
         lambda t: torch.softmax(t, -1), 0.0),
        ("log_softmax", lambda t: tt.log_softmax(t, -1),
         lambda t: torch.log_softmax(t, -1), 0.0),
    ],
)
def test_unary_op_parity(name, mine, theirs, offset):
    compare_unary(mine, theirs, randn(5, 4, seed=3) + offset)


def test_relu_subgradient_at_zero_matches_torch():
    """Both frameworks must choose relu'(0) = 0, not 0.5 or 1."""
    mine, theirs = pair([-1.0, 0.0, 1.0])
    tt.relu(mine).sum().backward()
    torch.relu(theirs).sum().backward()
    close(mine.grad, theirs.grad, what="relu subgradient")
    assert mine.grad[1] == 0.0


def test_maximum_tie_matches_torch():
    """torch.maximum splits a tie 0.5/0.5; TinyTorch must do the same."""
    a_mine, a_theirs = pair([1.0, 2.0])
    b_mine, b_theirs = pair([1.0, 1.0])
    tt.maximum(a_mine, b_mine).sum().backward()
    torch.maximum(a_theirs, b_theirs).sum().backward()
    close(a_mine.grad, a_theirs.grad, what="tie grad a")
    close(b_mine.grad, b_theirs.grad, what="tie grad b")


def test_extreme_logits_stay_finite_like_torch():
    """Stability check: log_softmax on huge logits must not overflow."""
    array = np.array([[1000.0, -1000.0, 0.0], [-800.0, 900.0, 850.0]])
    compare_unary(lambda t: tt.log_softmax(t, -1), lambda t: torch.log_softmax(t, -1), array)


# --------------------------------------------------------------------------
# Broadcasting.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [
        ((4, 3), (3,)),
        ((4, 1), (1, 3)),
        ((2, 4, 3), (3,)),
        ((2, 1, 3), (4, 3)),
        ((5, 1), (5, 4)),
        ((1,), (3, 4)),
    ],
)
def test_broadcast_parity(shape_a, shape_b):
    compare_binary(
        lambda a, b: a * b + a - b,
        lambda a, b: a * b + a - b,
        randn(*shape_a, seed=4),
        randn(*shape_b, seed=5),
    )


# --------------------------------------------------------------------------
# Matmul.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [
        ((4, 3), (3, 5)),
        ((3,), (3,)),
        ((4, 3), (3,)),
        ((3,), (3, 5)),
        ((2, 4, 3), (2, 3, 5)),
        ((2, 4, 3), (3, 5)),
        ((4, 3), (2, 3, 5)),
        ((3, 2, 4, 3), (3, 2, 3, 5)),
    ],
)
def test_matmul_parity(shape_a, shape_b):
    compare_binary(
        lambda a, b: a @ b, lambda a, b: a @ b, randn(*shape_a, seed=6), randn(*shape_b, seed=7)
    )


# --------------------------------------------------------------------------
# Reductions and shape ops.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum_parity(axis, keepdims):
    array = randn(3, 4, seed=8)
    kwargs_mine = {"axis": axis, "keepdims": keepdims}
    if axis is None:
        compare_unary(lambda t: tt.sum(t) * 2.0, lambda t: t.sum() * 2.0, array)
    else:
        compare_unary(
            lambda t: tt.sum(t, **kwargs_mine) * 2.0,
            lambda t: t.sum(dim=axis, keepdim=keepdims) * 2.0,
            array,
        )


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_mean_parity(axis):
    array = randn(3, 4, seed=9)
    if axis is None:
        compare_unary(lambda t: tt.mean(t) * 2.0, lambda t: t.mean() * 2.0, array)
    else:
        compare_unary(
            lambda t: tt.mean(t, axis=axis) * 2.0, lambda t: t.mean(dim=axis) * 2.0, array
        )


@pytest.mark.parametrize("axis", [0, 1])
def test_max_reduction_parity_uses_amax(axis):
    """TinyTorch's ``max`` splits ties, matching ``torch.amax`` (not
    ``torch.max``, which routes the whole gradient to one index)."""
    compare_unary(
        lambda t: tt.max(t, axis=axis) * 2.0,
        lambda t: torch.amax(t, dim=axis) * 2.0,
        randn(4, 5, seed=10),
    )


def test_reshape_parity():
    compare_unary(lambda t: t.reshape(4, 6) * 2.0, lambda t: t.reshape(4, 6) * 2.0, randn(2, 12, seed=11))


@pytest.mark.parametrize("axes", [(0, 1, 2), (2, 0, 1), (1, 0, 2)])
def test_transpose_parity(axes):
    compare_unary(
        lambda t: tt.transpose(t, axes) * 2.0,
        lambda t: t.permute(*axes) * 2.0,
        randn(2, 3, 4, seed=12),
    )


def test_indexing_parity():
    compare_unary(lambda t: t[1:3, ::2] * 2.0, lambda t: t[1:3, ::2] * 2.0, randn(5, 6, seed=13))


def test_fancy_indexing_parity_accumulates():
    rows = [0, 2, 2, 1]
    compare_unary(lambda t: t[rows] * 2.0, lambda t: t[rows] * 2.0, randn(4, 3, seed=14))


# --------------------------------------------------------------------------
# Losses.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_mse_loss_parity(reduction):
    compare_binary(
        lambda p, t: nn.MSELoss(reduction)(p, t),
        lambda p, t: torch.nn.MSELoss(reduction=reduction)(p, t),
        randn(6, 3, seed=15),
        randn(6, 3, seed=16),
    )


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_l1_loss_parity(reduction):
    compare_binary(
        lambda p, t: nn.L1Loss(reduction)(p, t),
        lambda p, t: torch.nn.L1Loss(reduction=reduction)(p, t),
        randn(6, 3, seed=17),
        randn(6, 3, seed=18) + 5.0,
    )


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cross_entropy_parity(reduction):
    labels = np.array([0, 3, 1, 2, 2, 0])
    logits_mine, logits_theirs = pair(randn(6, 4, seed=19))
    target_mine = tt.Tensor(labels)
    target_theirs = torch.tensor(labels, dtype=torch.long)

    loss_mine = nn.CrossEntropyLoss(reduction)(logits_mine, target_mine)
    loss_theirs = torch.nn.CrossEntropyLoss(reduction=reduction)(logits_theirs, target_theirs)
    close(loss_mine, loss_theirs, what="cross-entropy loss")

    loss_mine.backward()
    loss_theirs.backward()
    close(logits_mine.grad, logits_theirs.grad, what="cross-entropy gradient")


def test_cross_entropy_gradient_equals_softmax_minus_onehot():
    """The analytic identity, confirmed against both the engine and torch."""
    labels = np.array([1, 0, 2])
    logits_mine, logits_theirs = pair(randn(3, 4, seed=20))
    nn.CrossEntropyLoss()(logits_mine, tt.Tensor(labels)).backward()

    probs = torch.softmax(logits_theirs, dim=-1).detach().numpy()
    onehot = np.zeros_like(probs)
    onehot[np.arange(3), labels] = 1.0
    close(logits_mine.grad, (probs - onehot) / 3.0, what="softmax - onehot")


def test_nll_loss_parity():
    labels = np.array([2, 0, 1])
    mine, theirs = pair(randn(3, 4, seed=21))
    loss_mine = nn.NLLLoss()(tt.log_softmax(mine), tt.Tensor(labels))
    loss_theirs = torch.nn.NLLLoss()(torch.log_softmax(theirs, -1), torch.tensor(labels))
    close(loss_mine, loss_theirs, what="nll loss")
    loss_mine.backward()
    loss_theirs.backward()
    close(mine.grad, theirs.grad, what="nll gradient")


def test_bce_loss_parity():
    """Targets are constants here (a label is not a differentiation target)."""
    labels = np.array([0.0, 1.0, 1.0, 0.0])
    probs_mine, probs_theirs = pair([0.2, 0.6, 0.9, 0.4])
    target_mine = tt.Tensor(labels, requires_grad=False)
    target_theirs = torch.tensor(labels, dtype=torch.float64)

    loss_mine = nn.BCELoss()(probs_mine, target_mine)
    loss_theirs = torch.nn.BCELoss()(probs_theirs, target_theirs)
    close(loss_mine, loss_theirs, what="bce loss")

    loss_mine.backward()
    loss_theirs.backward()
    close(probs_mine.grad, probs_theirs.grad, what="bce gradient")


# --------------------------------------------------------------------------
# Layers and whole networks with weight transfer.
# --------------------------------------------------------------------------
def build_matched_mlp(sizes=(4, 8, 6, 3), activation="relu", seed=42):
    """Build the same MLP in both frameworks with bit-identical weights.

    Weights are drawn once in NumPy and copied into both models, so any
    difference in the outputs is attributable to the forward/backward maths and
    nothing else.
    """
    rng = np.random.default_rng(seed)
    act_mine = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "gelu": nn.GELU}[activation]
    act_theirs = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "gelu": torch.nn.GELU,
    }[activation]

    layers_mine, layers_theirs = [], []
    for i in range(len(sizes) - 1):
        layers_mine.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers_theirs.append(torch.nn.Linear(sizes[i], sizes[i + 1], dtype=torch.float64))
        if i < len(sizes) - 2:
            layers_mine.append(act_mine())
            layers_theirs.append(act_theirs())

    model_mine = nn.Sequential(*layers_mine)
    model_theirs = torch.nn.Sequential(*layers_theirs)

    with torch.no_grad():
        for lm, lt in zip(
            [m for m in layers_mine if isinstance(m, nn.Linear)],
            [m for m in layers_theirs if isinstance(m, torch.nn.Linear)],
            strict=True,
        ):
            w = rng.standard_normal(lm.weight.shape) * 0.5
            b = rng.standard_normal(lm.bias.shape) * 0.5
            lm.weight.data[...] = w
            lm.bias.data[...] = b
            lt.weight.copy_(torch.tensor(w, dtype=torch.float64))
            lt.bias.copy_(torch.tensor(b, dtype=torch.float64))
    return model_mine, model_theirs


def test_linear_layer_parity():
    layer_mine = nn.Linear(4, 3)
    layer_theirs = torch.nn.Linear(4, 3, dtype=torch.float64)
    with torch.no_grad():
        layer_theirs.weight.copy_(torch.tensor(layer_mine.weight.data))
        layer_theirs.bias.copy_(torch.tensor(layer_mine.bias.data))

    x_mine, x_theirs = pair(randn(6, 4, seed=22))
    out_mine, out_theirs = layer_mine(x_mine), layer_theirs(x_theirs)
    close(out_mine, out_theirs, what="linear forward")

    out_mine.sum().backward()
    out_theirs.sum().backward()
    close(layer_mine.weight.grad, layer_theirs.weight.grad, what="weight grad")
    close(layer_mine.bias.grad, layer_theirs.bias.grad, what="bias grad")
    close(x_mine.grad, x_theirs.grad, what="input grad")


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "gelu"])
def test_mlp_forward_and_gradient_parity(activation):
    """The completion test, parametrised over activations."""
    model_mine, model_theirs = build_matched_mlp(activation=activation)
    x_mine, x_theirs = pair(randn(16, 4, seed=23), requires_grad=False)
    labels = np.random.default_rng(24).integers(0, 3, size=16)

    loss_mine = nn.CrossEntropyLoss()(model_mine(x_mine), tt.Tensor(labels))
    loss_theirs = torch.nn.CrossEntropyLoss()(
        model_theirs(x_theirs), torch.tensor(labels, dtype=torch.long)
    )
    close(loss_mine, loss_theirs, what="loss")

    loss_mine.backward()
    loss_theirs.backward()
    for (name, p_mine), p_theirs in zip(
        model_mine.named_parameters(), model_theirs.parameters()
    , strict=True):
        close(p_mine.grad, p_theirs.grad, what=f"grad of {name}")


def test_float32_mlp_parity_at_relaxed_tolerance():
    """The same comparison in float32, at float32-appropriate tolerance.

    float32 is the default working precision for real training, so it is worth
    confirming parity survives it -- with tolerances that acknowledge NumPy and
    ATen accumulate reductions in different orders.
    """
    tt.set_default_dtype(np.float32)
    try:
        rng = np.random.default_rng(25)
        layer_mine = nn.Linear(6, 4)
        layer_theirs = torch.nn.Linear(6, 4, dtype=torch.float32)
        w = rng.standard_normal((4, 6)).astype(np.float32)
        b = rng.standard_normal(4).astype(np.float32)
        layer_mine.weight.data[...] = w
        layer_mine.bias.data[...] = b
        with torch.no_grad():
            layer_theirs.weight.copy_(torch.tensor(w))
            layer_theirs.bias.copy_(torch.tensor(b))

        x = rng.standard_normal((10, 6)).astype(np.float32)
        x_mine = tt.Tensor(x, requires_grad=True)
        x_theirs = torch.tensor(x, requires_grad=True)

        out_mine = tt.relu(layer_mine(x_mine)).sum()
        out_theirs = torch.relu(layer_theirs(x_theirs)).sum()
        close(out_mine, out_theirs, TORCH_F32_RTOL, TORCH_F32_ATOL, "float32 forward")

        out_mine.backward()
        out_theirs.backward()
        assert layer_mine.weight.grad.dtype == np.float32
        close(
            layer_mine.weight.grad, layer_theirs.weight.grad,
            TORCH_F32_RTOL, TORCH_F32_ATOL, "float32 weight grad",
        )
    finally:
        tt.set_default_dtype(np.float64)


# --------------------------------------------------------------------------
# Optimizers.
# --------------------------------------------------------------------------
def run_optimizer_parity(make_mine, make_theirs, steps=25, seed=30):
    """Train the same MLP with both optimizers and compare after every step.

    Comparing at *every* step rather than only at the end is what catches
    state bugs: a wrong bias-correction exponent or a momentum buffer seeded
    with zeros shows up on step 1 or 2 and then partially washes out.
    """
    model_mine, model_theirs = build_matched_mlp(seed=seed)
    opt_mine = make_mine(list(model_mine.parameters()))
    opt_theirs = make_theirs(list(model_theirs.parameters()))

    rng = np.random.default_rng(seed + 1)
    x = rng.standard_normal((12, 4))
    labels = rng.integers(0, 3, size=12)
    x_mine = tt.Tensor(x)
    x_theirs = torch.tensor(x, dtype=torch.float64)
    y_mine = tt.Tensor(labels)
    y_theirs = torch.tensor(labels, dtype=torch.long)

    for step in range(steps):
        opt_mine.zero_grad()
        loss_mine = nn.CrossEntropyLoss()(model_mine(x_mine), y_mine)
        loss_mine.backward()
        opt_mine.step()

        opt_theirs.zero_grad()
        loss_theirs = torch.nn.CrossEntropyLoss()(model_theirs(x_theirs), y_theirs)
        loss_theirs.backward()
        opt_theirs.step()

        close(loss_mine, loss_theirs, what=f"loss at step {step}")
        for (name, p_mine), p_theirs in zip(
            model_mine.named_parameters(), model_theirs.parameters()
        , strict=True):
            close(p_mine, p_theirs, what=f"{name} after step {step}")
    return model_mine, model_theirs, opt_mine, opt_theirs


def test_sgd_parity():
    run_optimizer_parity(
        lambda p: optim.SGD(p, lr=0.1),
        lambda p: torch.optim.SGD(p, lr=0.1),
    )


def test_sgd_momentum_parity():
    """Catches the classic bug of seeding the momentum buffer with zeros."""
    run_optimizer_parity(
        lambda p: optim.SGD(p, lr=0.05, momentum=0.9),
        lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    )


def test_sgd_nesterov_weight_decay_parity():
    run_optimizer_parity(
        lambda p: optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=1e-2, nesterov=True),
        lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=1e-2, nesterov=True),
    )


def test_sgd_dampening_parity():
    run_optimizer_parity(
        lambda p: optim.SGD(p, lr=0.05, momentum=0.9, dampening=0.2),
        lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, dampening=0.2),
    )


def test_adam_parity():
    """Catches bias-correction errors, which are largest on the first steps."""
    run_optimizer_parity(
        lambda p: optim.Adam(p, lr=1e-2),
        lambda p: torch.optim.Adam(p, lr=1e-2),
    )


def test_adam_weight_decay_parity():
    run_optimizer_parity(
        lambda p: optim.Adam(p, lr=1e-2, weight_decay=1e-2),
        lambda p: torch.optim.Adam(p, lr=1e-2, weight_decay=1e-2),
    )


def test_adam_amsgrad_parity():
    run_optimizer_parity(
        lambda p: optim.Adam(p, lr=1e-2, amsgrad=True),
        lambda p: torch.optim.Adam(p, lr=1e-2, amsgrad=True),
    )


def test_adam_custom_betas_parity():
    run_optimizer_parity(
        lambda p: optim.Adam(p, lr=5e-3, betas=(0.5, 0.99), eps=1e-9),
        lambda p: torch.optim.Adam(p, lr=5e-3, betas=(0.5, 0.99), eps=1e-9),
    )


def test_adamw_parity():
    run_optimizer_parity(
        lambda p: optim.AdamW(p, lr=1e-2, weight_decay=1e-2),
        lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=1e-2),
    )


def test_single_optimizer_step_parity_in_detail():
    """The completion-test requirement, isolated: identical weights, one
    forward, one backward, one step -- compared parameter by parameter."""
    model_mine, model_theirs = build_matched_mlp(seed=77)
    opt_mine = optim.Adam(model_mine.parameters(), lr=0.05)
    opt_theirs = torch.optim.Adam(model_theirs.parameters(), lr=0.05)

    x = np.random.default_rng(78).standard_normal((8, 4))
    target = np.random.default_rng(79).standard_normal((8, 3))

    loss_mine = nn.MSELoss()(model_mine(tt.Tensor(x)), tt.Tensor(target))
    loss_theirs = torch.nn.MSELoss()(
        model_theirs(torch.tensor(x, dtype=torch.float64)),
        torch.tensor(target, dtype=torch.float64),
    )
    close(loss_mine, loss_theirs, what="loss before step")

    opt_mine.zero_grad()
    loss_mine.backward()
    opt_theirs.zero_grad()
    loss_theirs.backward()
    for (name, p_mine), p_theirs in zip(model_mine.named_parameters(), model_theirs.parameters(), strict=True):
        close(p_mine.grad, p_theirs.grad, what=f"grad {name}")

    opt_mine.step()
    opt_theirs.step()
    for (name, p_mine), p_theirs in zip(model_mine.named_parameters(), model_theirs.parameters(), strict=True):
        close(p_mine, p_theirs, what=f"{name} after one Adam step")


def test_optimizer_state_survives_a_round_trip_mid_training():
    """Checkpoint Adam halfway, restore, and finish -- results must match a
    run that was never interrupted. This is what optimizer state is *for*."""
    model_a, _ = build_matched_mlp(seed=88)
    model_b, _ = build_matched_mlp(seed=88)
    x = tt.Tensor(np.random.default_rng(89).standard_normal((10, 4)))
    y = tt.Tensor(np.random.default_rng(90).integers(0, 3, size=10))

    opt_a = optim.Adam(model_a.parameters(), lr=1e-2)
    opt_b = optim.Adam(model_b.parameters(), lr=1e-2)

    def train(model, opt, steps):
        for _ in range(steps):
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            opt.step()

    train(model_a, opt_a, 10)

    train(model_b, opt_b, 5)
    checkpoint = {"model": model_b.state_dict(), "optim": opt_b.state_dict()}
    model_c, _ = build_matched_mlp(seed=88)
    opt_c = optim.Adam(model_c.parameters(), lr=1e-2)
    model_c.load_state_dict(checkpoint["model"])
    opt_c.load_state_dict(checkpoint["optim"])
    train(model_c, opt_c, 5)

    for (name, p_a), (_, p_c) in zip(model_a.named_parameters(), model_c.named_parameters(), strict=True):
        close(p_c, p_a.data, what=f"{name} after resumed training")
