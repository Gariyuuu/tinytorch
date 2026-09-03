"""Autograd engine semantics: graph shape, accumulation, detach, grad mode.

These tests are about the *engine*, not about any particular derivative
formula; the formulas are checked numerically in ``test_gradcheck.py`` and
against PyTorch in ``test_torch_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytorch as tt
from tinytorch.autograd import topological_order, unbroadcast

from .conftest import tensor64


# --------------------------------------------------------------------------
# Scalar autodiff: the base case, checked by hand.
# --------------------------------------------------------------------------
def test_scalar_chain_rule_by_hand():
    """d/dx [ (3x + 2)^2 ] = 6(3x + 2); at x = 2 that is 48."""
    x = tensor64(2.0)
    y = (3.0 * x + 2.0) ** 2
    y.backward()
    assert y.item() == pytest.approx(64.0)
    assert x.grad == pytest.approx(48.0)


def test_scalar_product_and_quotient_rules():
    """f = (x*y)/(x+y); check both partials against the closed form."""
    x, y = tensor64(3.0), tensor64(5.0)
    f = (x * y) / (x + y)
    f.backward()
    # df/dx = y^2 / (x+y)^2, df/dy = x^2 / (x+y)^2
    assert x.grad == pytest.approx(25.0 / 64.0)
    assert y.grad == pytest.approx(9.0 / 64.0)


def test_exp_log_inverse_gradient():
    """log(exp(x)) is the identity, so its derivative must be exactly 1."""
    x = tensor64([0.5, -2.0, 3.0])
    tt.log(tt.exp(x)).sum().backward()
    np.testing.assert_allclose(x.grad, np.ones(3), rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------
# Branched graphs and tensor reuse.
# --------------------------------------------------------------------------
def test_branched_graph_sums_both_paths():
    """A tensor feeding two branches receives the sum of both gradients.

    y = x^2 + 3x  =>  dy/dx = 2x + 3.
    """
    x = tensor64([1.0, 2.0, 3.0])
    y = (x * x + 3.0 * x).sum()
    y.backward()
    np.testing.assert_allclose(x.grad, 2 * np.array([1.0, 2.0, 3.0]) + 3.0)


def test_diamond_graph():
    """A diamond (a -> b, a -> c, (b, c) -> d) must visit `a` exactly once
    and only after both branches have contributed."""
    a = tensor64([2.0, 3.0])
    b = a * 4.0
    c = a + 10.0
    d = (b * c).sum()
    d.backward()
    # d = 4a(a+10) => dd/da = 8a + 40
    np.testing.assert_allclose(a.grad, 8 * np.array([2.0, 3.0]) + 40.0)
    # `a` appears once in the traversal despite two consumers.
    order = list(topological_order(d))
    assert sum(1 for t in order if t is a) == 1


def test_deeply_reused_tensor():
    """x used n times in a product: d/dx x^n = n x^(n-1)."""
    x = tensor64(1.5)
    out = x
    for _ in range(9):
        out = out * x
    out.backward()
    assert out.item() == pytest.approx(1.5**10)
    assert x.grad == pytest.approx(10 * 1.5**9)


def test_deep_chain_does_not_recurse():
    """A 5000-op chain must not blow the Python recursion limit.

    This is why the traversal is an explicit stack rather than a recursive
    walk -- CPython's default limit is 1000 frames.
    """
    x = tensor64(0.0)
    out = x
    for _ in range(5000):
        out = out + 1.0
    out.backward()
    assert out.item() == pytest.approx(5000.0)
    assert x.grad == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Gradient accumulation.
# --------------------------------------------------------------------------
def test_gradient_accumulates_across_backward_calls():
    """Two backward passes without zero_grad sum -- this is intentional."""
    x = tensor64([1.0, 2.0])
    (x * x).sum().backward()
    first = x.grad.copy()
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * first)


def test_zero_grad_resets_accumulation():
    x = tensor64([1.0, 2.0])
    (x * x).sum().backward()
    x.zero_grad()
    assert x.grad is None
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad, [2.0, 4.0])


def test_microbatch_accumulation_equals_full_batch():
    """Summing gradients over micro-batches equals one full-batch gradient.

    This is the property the whole gradient-accumulation training trick rests
    on, so it deserves a direct test.
    """
    data = np.arange(12, dtype=np.float64).reshape(6, 2)

    w_full = tensor64([[1.0], [2.0]])
    full = ((tt.Tensor(data) @ w_full) ** 2).sum()
    full.backward()

    w_micro = tensor64([[1.0], [2.0]])
    for start in (0, 2, 4):
        chunk = tt.Tensor(data[start : start + 2])
        ((chunk @ w_micro) ** 2).sum().backward()

    np.testing.assert_allclose(w_micro.grad, w_full.grad, rtol=1e-12)


def test_zero_grad_set_to_none_false_keeps_zeros():
    x = tensor64([1.0, 2.0])
    (x * x).sum().backward()
    x.zero_grad(set_to_none=False)
    np.testing.assert_allclose(x.grad, np.zeros(2))


# --------------------------------------------------------------------------
# requires_grad, leaves and detach.
# --------------------------------------------------------------------------
def test_no_graph_without_requires_grad():
    x = tt.Tensor([1.0, 2.0], requires_grad=False)
    y = x * 3.0
    assert y.grad_fn is None
    assert not y.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        y.sum().backward()


def test_requires_grad_propagates_through_ops():
    a = tensor64([1.0])
    b = tt.Tensor([2.0], requires_grad=False)
    out = a * b
    assert out.requires_grad and out.grad_fn is not None


def test_detach_cuts_the_graph_and_shares_storage():
    x = tensor64([1.0, 2.0, 3.0])
    y = (x * 2.0).detach()
    assert y.grad_fn is None and not y.requires_grad and y.is_leaf
    # Detaching mid-graph stops gradient flow entirely.
    z = tensor64([4.0])
    ((x * 2.0).detach() * z).sum().backward()
    assert x.grad is None and z.grad is not None
    # Storage is shared with the source, not copied.
    d = x.detach()
    d.data[0] = 99.0
    assert x.data[0] == 99.0


def test_non_leaf_grad_requires_retain_grad():
    x = tensor64([2.0])
    mid = x * 3.0
    out = (mid * mid).sum()
    out.backward()
    assert mid.grad is None, "intermediate tensors should not hold gradient by default"

    x2 = tensor64([2.0])
    mid2 = (x2 * 3.0).retain_grad()
    ((mid2 * mid2).sum()).backward()
    np.testing.assert_allclose(mid2.grad, [12.0])  # d(m^2)/dm at m=6


def test_integer_tensor_cannot_require_grad():
    with pytest.raises(RuntimeError, match="only floating tensors"):
        tt.Tensor([1, 2, 3], requires_grad=True)


def test_clearing_requires_grad_on_non_leaf_is_rejected():
    x = tensor64([1.0])
    y = x * 2.0
    with pytest.raises(RuntimeError, match="non-leaf"):
        y.requires_grad = False


# --------------------------------------------------------------------------
# Grad mode.
# --------------------------------------------------------------------------
def test_no_grad_disables_recording():
    x = tensor64([1.0, 2.0])
    with tt.no_grad():
        y = x * 2.0
        assert not y.requires_grad and y.grad_fn is None
    assert tt.is_grad_enabled()
    assert (x * 2.0).requires_grad


def test_no_grad_restores_previous_state_on_exception():
    with pytest.raises(ValueError), tt.no_grad():
        raise ValueError("boom")
    assert tt.is_grad_enabled()


def test_enable_grad_inside_no_grad():
    x = tensor64([1.0])
    with tt.no_grad(), tt.enable_grad():
        assert (x * 2.0).requires_grad


def test_no_grad_as_decorator():
    @tt.no_grad()
    def infer(t):
        return t * 2.0

    assert not infer(tensor64([1.0])).requires_grad
    assert tt.is_grad_enabled()


# --------------------------------------------------------------------------
# Backward seeds and error handling.
# --------------------------------------------------------------------------
def test_non_scalar_backward_requires_explicit_gradient():
    x = tensor64([1.0, 2.0])
    with pytest.raises(RuntimeError, match="scalar outputs"):
        (x * 2.0).backward()


def test_explicit_seed_is_a_vector_jacobian_product():
    """backward(v) computes v^T J, not the full Jacobian."""
    x = tensor64([1.0, 2.0, 3.0])
    y = x * x  # J = diag(2x)
    y.backward([1.0, 0.0, 10.0])
    np.testing.assert_allclose(x.grad, [2.0, 0.0, 60.0])


def test_mismatched_seed_shape_is_rejected():
    x = tensor64([1.0, 2.0])
    with pytest.raises(RuntimeError, match="does not match"):
        (x * 2.0).backward([1.0, 2.0, 3.0])


# --------------------------------------------------------------------------
# Broadcast gradient reduction.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("grad_shape", "target_shape"),
    [
        ((4, 3), (3,)),
        ((4, 3), (1, 3)),
        ((4, 3), (4, 1)),
        ((2, 4, 3), (3,)),
        ((2, 4, 3), (4, 3)),
        ((2, 4, 3), (1, 1, 1)),
        ((2, 4, 3), ()),
        ((3,), (3,)),
    ],
)
def test_unbroadcast_shapes_and_sums(grad_shape, target_shape):
    """unbroadcast must both fix the shape and preserve the total sum."""
    grad = np.arange(int(np.prod(grad_shape)), dtype=np.float64).reshape(grad_shape)
    reduced = unbroadcast(grad, target_shape)
    assert reduced.shape == target_shape
    assert reduced.sum() == pytest.approx(grad.sum())


def test_broadcast_bias_gradient_is_the_batch_sum():
    """The canonical case: a bias broadcast over a batch of 5 rows."""
    bias = tensor64([1.0, 2.0, 3.0])
    x = tt.Tensor(np.ones((5, 3)))
    (x + bias).sum().backward()
    np.testing.assert_allclose(bias.grad, [5.0, 5.0, 5.0])


def test_broadcast_both_operands():
    a = tensor64(np.ones((4, 1)))
    b = tensor64(np.ones((1, 3)))
    (a * b).sum().backward()
    np.testing.assert_allclose(a.grad, np.full((4, 1), 3.0))
    np.testing.assert_allclose(b.grad, np.full((1, 3), 4.0))


# --------------------------------------------------------------------------
# Function contract.
# --------------------------------------------------------------------------
def test_function_returning_wrong_arity_is_caught():
    """A Function must return one gradient per positional input."""

    class Broken(tt.Function):
        @staticmethod
        def forward(ctx, a, b):
            return a + b

        @staticmethod
        def backward(ctx, g):
            return (g,)  # missing the second gradient

    out = Broken.apply(tensor64([1.0]), tensor64([2.0]))
    with pytest.raises(RuntimeError, match="returned 1 gradient"):
        out.sum().backward()


def test_custom_function_integrates_with_the_engine():
    """A user-defined Function participates like any built-in op."""

    class Square(tt.Function):
        @staticmethod
        def forward(ctx, a):
            ctx.save_for_backward(a)
            return a * a

        @staticmethod
        def backward(ctx, g):
            (a,) = ctx.saved_tensors
            return 2 * a * g

    x = tensor64([3.0, -4.0])
    (Square.apply(x) * 2.0).sum().backward()
    np.testing.assert_allclose(x.grad, [12.0, -16.0])
