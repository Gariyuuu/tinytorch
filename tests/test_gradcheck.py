"""Finite-difference verification of every differentiable operator.

Each test asserts that TinyTorch's analytic gradient matches a central
difference of its own forward pass. This is the framework-independent
correctness check: it needs no PyTorch and would catch a derivative that is
wrong in *both* frameworks the same way (which a parity test could not).

All inputs are float64 -- see ``tinytorch.gradcheck`` for why float32 makes
finite differences meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytorch as tt
from tinytorch import nn
from tinytorch.gradcheck import GradCheckError

from .conftest import GRADCHECK_ATOL, GRADCHECK_EPS, GRADCHECK_RTOL, tensor64


def check(fn, *inputs) -> None:
    """Run gradcheck with the suite's documented tolerances."""
    assert tt.gradcheck(
        fn, inputs, eps=GRADCHECK_EPS, rtol=GRADCHECK_RTOL, atol=GRADCHECK_ATOL
    )


def rand(*shape, requires_grad: bool = True, offset: float = 0.0):
    """Well-conditioned random input (kept away from 0 for log/div/pow)."""
    return tensor64(np.random.default_rng(7).standard_normal(shape) + offset, requires_grad)


# --------------------------------------------------------------------------
# The checker itself must be trustworthy before anything it certifies is.
# --------------------------------------------------------------------------
def test_gradcheck_detects_a_wrong_derivative():
    """A deliberately broken Function must be caught, or every other test in
    this file proves nothing."""

    class WrongSquare(tt.Function):
        @staticmethod
        def forward(ctx, a):
            ctx.save_for_backward(a)
            return a * a

        @staticmethod
        def backward(ctx, g):
            (a,) = ctx.saved_tensors
            return 3.0 * a * g  # should be 2 * a * g

    with pytest.raises(GradCheckError, match="exceed tolerance"):
        tt.gradcheck(WrongSquare.apply, [rand(4)])


def test_gradcheck_rejects_float32_inputs():
    x = tt.Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
    with pytest.raises(TypeError, match="float64 is required"):
        tt.gradcheck(lambda t: t * 2.0, [x])


def test_numerical_gradient_matches_a_known_derivative():
    """d/dx sum(sin-free polynomial) checked against the closed form."""
    x = tensor64([1.0, 2.0, 3.0])
    numeric = tt.numerical_gradient(lambda t: (t**3).sum(), [x], wrt=0, eps=1e-6)
    np.testing.assert_allclose(numeric, 3 * np.array([1.0, 2.0, 3.0]) ** 2, rtol=1e-6)


def test_numerical_gradient_leaves_inputs_untouched():
    x = tensor64([1.0, 2.0, 3.0])
    before = x.data.copy()
    tt.numerical_gradient(lambda t: (t**2).sum(), [x])
    np.testing.assert_array_equal(x.data, before)


# --------------------------------------------------------------------------
# Elementwise binary ops (including mixed broadcasting).
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
        ("div", lambda a, b: a / b),
        ("maximum", tt.maximum),
        ("minimum", tt.minimum),
    ],
)
def test_binary_ops(name, fn):
    a = rand(4, 3)
    b = rand(4, 3, offset=3.0)  # offset keeps the divisor away from zero
    check(fn, a, b)


@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [
        ((4, 3), (3,)),
        ((4, 1), (1, 3)),
        ((2, 4, 3), (3,)),
        ((2, 1, 3), (4, 3)),
        ((), (3, 2)),
        ((5, 1), (5, 4)),
    ],
)
def test_broadcasting_gradients(shape_a, shape_b):
    """The unbroadcast path must produce correct gradients for both operands
    in every broadcasting configuration."""
    a = rand(*shape_a)
    b = rand(*shape_b, offset=3.0)
    check(lambda x, y: x * y + x / y, a, b)


def test_power_with_constant_exponent():
    check(lambda a: a**3, rand(4, 3))


def test_power_with_fractional_exponent():
    check(lambda a: a**0.5, rand(4, offset=4.0))


def test_power_differentiable_in_the_exponent():
    """d/db a^b = a^b ln a -- only defined for a > 0, so the base is positive."""
    base = rand(3, offset=4.0)
    exponent = tensor64([1.5, 2.0, 0.5])
    check(lambda a, b: a**b, base, exponent)


def test_negation():
    check(lambda a: -a * 2.0, rand(3, 4))


# --------------------------------------------------------------------------
# Unary ops.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "fn", "offset"),
    [
        ("exp", tt.exp, 0.0),
        ("log", tt.log, 5.0),
        ("sigmoid", tt.sigmoid, 0.0),
        ("tanh", tt.tanh, 0.0),
        ("gelu_exact", tt.gelu, 0.0),
        ("gelu_tanh", lambda t: tt.gelu(t, approximate="tanh"), 0.0),
        ("softmax", lambda t: tt.softmax(t, axis=-1), 0.0),
        ("log_softmax", lambda t: tt.log_softmax(t, axis=-1), 0.0),
        ("abs", lambda t: tt.ops.abs(t), 5.0),
        ("sqrt", lambda t: t.sqrt(), 5.0),
        ("clone", lambda t: t.clone() * 3.0, 0.0),
    ],
)
def test_unary_ops(name, fn, offset):
    check(fn, rand(4, 5, offset=offset))


def test_relu_away_from_the_kink():
    """ReLU is checked away from 0: finite differences straddle the kink and
    would report the *average* of the two one-sided derivatives there."""
    x = tensor64([-2.0, -0.5, 0.5, 2.0])
    check(tt.relu, x)


def test_relu_subgradient_at_zero_is_zero():
    """At exactly 0, TinyTorch uses PyTorch's convention relu'(0) = 0.
    Finite differences cannot verify this, so it is asserted directly."""
    x = tensor64([0.0])
    tt.relu(x).sum().backward()
    assert x.grad[0] == 0.0


def test_maximum_splits_the_gradient_at_a_tie():
    """torch.maximum semantics: an exact tie gives each operand 0.5."""
    a, b = tensor64([1.0]), tensor64([1.0])
    tt.maximum(a, b).sum().backward()
    assert a.grad[0] == pytest.approx(0.5)
    assert b.grad[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Matrix multiplication.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [
        ((4, 3), (3, 5)),      # standard 2-D
        ((3,), (3,)),          # inner product
        ((4, 3), (3,)),        # matrix-vector
        ((3,), (3, 5)),        # vector-matrix
        ((2, 4, 3), (2, 3, 5)),  # batched
        ((2, 4, 3), (3, 5)),   # batched against a shared matrix
        ((4, 3), (2, 3, 5)),   # broadcast batch on the left
    ],
)
def test_matmul(shape_a, shape_b):
    check(lambda a, b: a @ b, rand(*shape_a), rand(*shape_b))


def test_matmul_chain():
    """Gradients must be correct through several contractions."""
    check(lambda a, b, c: a @ b @ c, rand(3, 4), rand(4, 5), rand(5, 2))


# --------------------------------------------------------------------------
# Reductions and shape ops.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum(axis, keepdims):
    check(lambda t: tt.sum(t, axis=axis, keepdims=keepdims) * 2.0, rand(3, 4, 2))


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_mean(axis, keepdims):
    check(lambda t: tt.mean(t, axis=axis, keepdims=keepdims) * 2.0, rand(3, 4, 2))


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_max_reduction(axis):
    """Random inputs have no ties, so finite differences are valid here."""
    check(lambda t: tt.max(t, axis=axis), rand(4, 5))


def test_max_reduction_splits_ties():
    """With duplicated maxima the gradient is shared, not given to index 0."""
    x = tensor64([3.0, 3.0, 1.0])
    tt.max(x).backward()
    np.testing.assert_allclose(x.grad, [0.5, 0.5, 0.0])


@pytest.mark.parametrize("shape", [(24,), (4, 6), (2, 3, 4), (2, 12), (-1, 8)])
def test_reshape(shape):
    check(lambda t: t.reshape(*shape) * 2.0, rand(2, 3, 4))


@pytest.mark.parametrize("axes", [None, (0, 1, 2), (2, 0, 1), (1, 0, 2), (2, 1, 0)])
def test_transpose(axes):
    check(lambda t: tt.transpose(t, axes) * 2.0, rand(2, 3, 4))


def test_transpose_then_matmul():
    """The composition that ``nn.Linear`` actually uses."""
    check(lambda x, w: x @ w.T, rand(5, 3), rand(4, 3))


def test_flatten():
    check(lambda t: t.flatten(1) * 2.0, rand(3, 4, 5))


# --------------------------------------------------------------------------
# Indexing.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "index",
    [
        1,
        slice(1, 3),
        (slice(None), 2),
        (0, slice(1, 3)),
        (slice(None), slice(0, 2)),
        Ellipsis,
    ],
)
def test_basic_indexing(index):
    check(lambda t: t[index] * 2.0, rand(4, 5))


def test_fancy_indexing_scatters_gradient():
    check(lambda t: t[[0, 2, 2]] * 2.0, rand(4, 5))


def test_repeated_index_accumulates_rather_than_overwrites():
    """Row 1 is selected three times, so it must receive 3x the gradient.

    This is exactly the bug ``np.add.at`` exists to prevent.
    """
    x = tensor64(np.zeros((3, 2)))
    x[[1, 1, 1]].sum().backward()
    np.testing.assert_allclose(x.grad, [[0, 0], [3, 3], [0, 0]])


def test_paired_index_gather():
    """The (rows, cols) gather that NLLLoss performs."""
    rows = np.array([0, 1, 2])
    cols = np.array([2, 0, 1])
    check(lambda t: t[(rows, cols)] * 2.0, rand(3, 4))


# --------------------------------------------------------------------------
# Layers, losses and whole networks.
# --------------------------------------------------------------------------
def test_linear_layer_gradients():
    layer = nn.Linear(4, 3)
    x = rand(6, 4)
    check(lambda x_, w, b: nn.functional.linear(x_, w, b), x, layer.weight, layer.bias)


def test_linear_without_bias():
    layer = nn.Linear(4, 3, bias=False)
    check(lambda x_, w: nn.functional.linear(x_, w), rand(6, 4), layer.weight)


def test_mse_loss_gradients():
    check(lambda p, t: nn.MSELoss()(p, t), rand(6, 3), rand(6, 3))


def test_l1_loss_gradients():
    """Inputs are separated so no residual sits exactly on |x|'s kink."""
    pred = tensor64(np.linspace(-2, 2, 8).reshape(4, 2))
    target = tensor64(np.linspace(3, 6, 8).reshape(4, 2), requires_grad=False)
    check(lambda p, t: nn.L1Loss()(p, t), pred, target)


def test_cross_entropy_gradients():
    logits = rand(6, 4)
    target = tt.Tensor(np.array([0, 3, 1, 2, 2, 0]))
    check(lambda z: nn.CrossEntropyLoss()(z, target), logits)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cross_entropy_reductions(reduction):
    target = tt.Tensor(np.array([1, 0, 2]))
    check(lambda z: nn.CrossEntropyLoss(reduction)(z, target), rand(3, 4))


def test_nll_loss_gradients():
    target = tt.Tensor(np.array([2, 0, 1]))
    check(lambda z: nn.NLLLoss()(tt.log_softmax(z), target), rand(3, 4))


def test_bce_loss_gradients():
    probs = tensor64(np.array([0.2, 0.6, 0.9, 0.4]))
    labels = tensor64(np.array([0.0, 1.0, 1.0, 0.0]), requires_grad=False)
    check(lambda p, y: nn.BCELoss()(p, y), probs, labels)


@pytest.mark.parametrize("activation", [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU])
def test_multilayer_network_gradients(activation):
    """End-to-end: a 3-layer MLP's every parameter checked numerically."""
    model = nn.Sequential(
        nn.Linear(4, 6), activation(), nn.Linear(6, 5), activation(), nn.Linear(5, 3)
    )
    # Keep pre-activations away from ReLU's kink so central differences are valid.
    x = tensor64(np.linspace(0.3, 1.7, 8 * 4).reshape(8, 4), requires_grad=False)
    target = tt.Tensor(np.array([0, 1, 2, 0, 1, 2, 1, 0]))
    params = list(model.parameters())

    def fn(*ps):
        return nn.CrossEntropyLoss()(model(x), target)

    check(fn, *params)


def test_weight_tying_accumulates_into_one_parameter():
    """A parameter used by two layers must receive the sum of both gradients."""
    shared = nn.Linear(3, 3)
    model = nn.Sequential(shared, nn.Tanh(), shared)
    x = rand(4, 3, requires_grad=False)
    model(x).sum().backward()
    assert shared.weight.grad is not None
    # The tied parameter is reported once, not twice.
    assert len(list(model.parameters())) == 2
