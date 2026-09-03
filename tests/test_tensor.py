"""Tensor semantics: construction, dtype policy, views, conversion, repr."""

from __future__ import annotations

import numpy as np
import pytest

import tinytorch as tt


# --------------------------------------------------------------------------
# Construction and dtype.
# --------------------------------------------------------------------------
def test_python_scalars_and_lists_use_the_default_dtype():
    tt.set_default_dtype(np.float32)
    try:
        assert tt.Tensor(1.0).dtype == np.float32
        assert tt.Tensor([1.0, 2.0]).dtype == np.float32
    finally:
        tt.set_default_dtype(np.float64)


def test_an_explicit_ndarray_keeps_its_own_dtype():
    """An incoming array is authoritative: silently narrowing a float64 array
    to the float32 default would destroy precision the caller asked for."""
    tt.set_default_dtype(np.float32)
    try:
        assert tt.Tensor(np.ones(3, dtype=np.float64)).dtype == np.float64
        assert tt.Tensor(np.ones(3, dtype=np.float32)).dtype == np.float32
    finally:
        tt.set_default_dtype(np.float64)


def test_explicit_dtype_wins():
    assert tt.Tensor([1.0], dtype=np.float32).dtype == np.float32


def test_integer_tensors_are_allowed_but_not_differentiable():
    t = tt.Tensor([1, 2, 3])
    assert t.dtype.kind == "i" and not t.requires_grad


def test_set_default_dtype_rejects_non_float():
    with pytest.raises(TypeError, match="floating type"):
        tt.set_default_dtype(np.int32)


def test_construction_copies_by_default():
    array = np.ones(3)
    t = tt.Tensor(array)
    t.data[0] = 5.0
    assert array[0] == 1.0


def test_from_numpy_shares_storage():
    array = np.ones(3)
    t = tt.from_numpy(array)
    t.data[0] = 5.0
    assert array[0] == 5.0


def test_tensor_from_tensor():
    source = tt.Tensor([1.0, 2.0], requires_grad=True)
    assert not tt.Tensor(source).requires_grad
    np.testing.assert_array_equal(tt.Tensor(source).data, source.data)


# --------------------------------------------------------------------------
# Properties.
# --------------------------------------------------------------------------
def test_shape_ndim_size():
    t = tt.zeros(2, 3, 4)
    assert t.shape == (2, 3, 4) and t.ndim == 3 and t.size == 24
    assert len(t) == 2


def test_is_leaf():
    x = tt.Tensor([1.0], requires_grad=True)
    assert x.is_leaf and not (x * 2.0).is_leaf


def test_item_requires_a_single_element():
    assert tt.Tensor([3.5]).item() == 3.5
    with pytest.raises(ValueError, match="single-element"):
        tt.Tensor([1.0, 2.0]).item()


def test_bool_of_multielement_tensor_is_ambiguous():
    assert bool(tt.Tensor([1.0]))
    with pytest.raises(RuntimeError, match="ambiguous"):
        bool(tt.Tensor([1.0, 2.0]))


def test_len_of_scalar_is_an_error():
    with pytest.raises(TypeError, match="0-d"):
        len(tt.Tensor(1.0))


def test_iteration_yields_rows():
    rows = list(tt.Tensor(np.arange(6.0).reshape(3, 2)))
    assert len(rows) == 3
    np.testing.assert_allclose(rows[1].data, [2.0, 3.0])


# --------------------------------------------------------------------------
# Creation helpers.
# --------------------------------------------------------------------------
def test_creation_helpers_accept_both_shape_forms():
    assert tt.zeros(2, 3).shape == tt.zeros((2, 3)).shape == (2, 3)


def test_creation_helpers_values():
    np.testing.assert_array_equal(tt.zeros(2, 2).data, np.zeros((2, 2)))
    np.testing.assert_array_equal(tt.ones(2, 2).data, np.ones((2, 2)))
    np.testing.assert_array_equal(tt.full((2,), 3.0).data, [3.0, 3.0])
    np.testing.assert_array_equal(tt.eye(3).data, np.eye(3))
    np.testing.assert_array_equal(tt.arange(4).data, [0, 1, 2, 3])
    np.testing.assert_allclose(tt.linspace(0, 1, 3).data, [0.0, 0.5, 1.0])


def test_zeros_like_and_ones_like_match_dtype_and_shape():
    source = tt.Tensor(np.ones((2, 3), dtype=np.float32))
    assert tt.zeros_like(source).dtype == np.float32
    assert tt.ones_like(source).shape == (2, 3)


def test_manual_seed_makes_random_creation_reproducible():
    tt.manual_seed(3)
    a = tt.randn(4, 4).data.copy()
    tt.manual_seed(3)
    np.testing.assert_array_equal(tt.randn(4, 4).data, a)


def test_rand_is_in_the_unit_interval():
    values = tt.rand(1000).data
    assert values.min() >= 0.0 and values.max() < 1.0


# --------------------------------------------------------------------------
# Operator overloads and reflected forms.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (lambda t: t + 1.0, [2.0, 3.0]),
        (lambda t: 1.0 + t, [2.0, 3.0]),
        (lambda t: t - 1.0, [0.0, 1.0]),
        (lambda t: 5.0 - t, [4.0, 3.0]),
        (lambda t: t * 3.0, [3.0, 6.0]),
        (lambda t: 3.0 * t, [3.0, 6.0]),
        (lambda t: t / 2.0, [0.5, 1.0]),
        (lambda t: 4.0 / t, [4.0, 2.0]),
        (lambda t: t**2, [1.0, 4.0]),
        (lambda t: 2.0**t, [2.0, 4.0]),
        (lambda t: -t, [-1.0, -2.0]),
    ],
)
def test_arithmetic_overloads(expression, expected):
    np.testing.assert_allclose(expression(tt.Tensor([1.0, 2.0])).data, expected)


def test_matmul_overload_and_reflected_form():
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(3.0).reshape(3, 1)
    np.testing.assert_allclose((tt.Tensor(a) @ tt.Tensor(b)).data, a @ b)
    np.testing.assert_allclose((a @ tt.Tensor(b)).data, a @ b)


def test_comparison_operators_return_boolean_tensors_without_grad():
    a = tt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = a > 1.5
    assert result.dtype == np.bool_ and not result.requires_grad
    np.testing.assert_array_equal(result.data, [False, True, True])
    np.testing.assert_array_equal((a == tt.Tensor([1.0, 0.0, 3.0])).data, [True, False, True])


def test_transpose_property():
    a = np.arange(6.0).reshape(2, 3)
    np.testing.assert_allclose(tt.Tensor(a).T.data, a.T)


def test_view_is_reshape():
    assert tt.zeros(2, 6).view(3, 4).shape == (3, 4)
    assert tt.zeros(2, 3, 4).flatten(1).shape == (2, 12)


def test_method_and_functional_forms_agree():
    t = tt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(t.sum().data, tt.sum(t).data)
    np.testing.assert_allclose(t.mean(axis=0).data, tt.mean(t, axis=0).data)
    np.testing.assert_allclose(t.exp().data, tt.exp(t).data)
    np.testing.assert_allclose(t.relu().data, tt.relu(t).data)
    np.testing.assert_allclose(t.softmax().data, tt.softmax(t).data)


def test_argmax_and_max():
    t = tt.Tensor([[1.0, 5.0], [7.0, 2.0]])
    np.testing.assert_array_equal(t.argmax(axis=1).data, [1, 0])
    assert t.max().item() == 7.0


# --------------------------------------------------------------------------
# Conversion.
# --------------------------------------------------------------------------
def test_numpy_returns_the_underlying_buffer():
    t = tt.Tensor([1.0, 2.0])
    view = t.numpy()
    view[0] = 9.0
    assert t.data[0] == 9.0


def test_np_asarray_interop():
    t = tt.Tensor([1.0, 2.0])
    np.testing.assert_allclose(np.asarray(t), [1.0, 2.0])
    assert float(tt.Tensor([2.5])) == 2.5
    assert tt.Tensor([[1.0, 2.0]]).tolist() == [[1.0, 2.0]]


def test_astype_is_detached():
    t = tt.Tensor([1.0, 2.0], requires_grad=True)
    cast = t.astype(np.float32)
    assert cast.dtype == np.float32 and not cast.requires_grad


def test_clone_is_differentiable_and_copies_storage():
    x = tt.Tensor([1.0, 2.0], requires_grad=True)
    c = x.clone()
    c.sum().backward()
    np.testing.assert_allclose(x.grad, [1.0, 1.0])
    c.data[0] = 99.0
    assert x.data[0] == 1.0


def test_copy_in_place_checks_shape():
    t = tt.Tensor(np.zeros(3))
    t.copy_(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(t.data, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shape mismatch"):
        t.copy_(np.zeros(4))


# --------------------------------------------------------------------------
# repr.
# --------------------------------------------------------------------------
def test_repr_shows_grad_fn_for_non_leaves():
    x = tt.Tensor([1.0], requires_grad=True)
    assert "requires_grad=True" in repr(x)
    assert "grad_fn=<Mul>" in repr(x * 2.0)


def test_parameter_repr_is_labelled():
    from tinytorch.nn import Parameter

    assert repr(Parameter(np.zeros(2))).startswith("Parameter containing:")


def test_unsupported_dtype_is_rejected():
    with pytest.raises(TypeError, match="unsupported dtype"):
        tt.Tensor(np.array(["a", "b"]))
