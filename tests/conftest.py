"""Shared fixtures and tolerance policy for the TinyTorch test suite.

Tolerance policy
----------------
Three regimes, each with a documented reason for its numbers.

``GRADCHECK_*`` (float64, finite differences)
    ``rtol=1e-5``. The central difference has error
    ``O(h^2) + O(eps_machine / h)``; at ``h=1e-6`` in float64 that floor is
    about ``1e-10`` absolute, but relative error on entries whose gradient is
    near zero is dominated by the ``atol`` term, so we keep ``atol=1e-8``.

``TORCH_F64_*`` (float64, TinyTorch vs PyTorch)
    ``rtol=1e-10, atol=1e-12``. Both frameworks run the *same* IEEE-754
    double-precision kernels, so any disagreement beyond a handful of ulps
    across a few dozen operations means a genuine formula difference, not
    rounding. This is the strict comparison and the one that actually proves
    correctness.

``TORCH_F32_*`` (float32, TinyTorch vs PyTorch)
    ``rtol=1e-5, atol=1e-6``. float32 carries ~7 decimal digits, and
    accumulation order inside BLAS differs between NumPy and ATen, so
    reductions legitimately diverge in the last couple of digits. Anything
    tighter would be testing BLAS, not TinyTorch.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytorch as tt

# -- documented tolerances --------------------------------------------------
GRADCHECK_EPS = 1e-6
GRADCHECK_RTOL = 1e-5
GRADCHECK_ATOL = 1e-8

TORCH_F64_RTOL = 1e-10
TORCH_F64_ATOL = 1e-12

TORCH_F32_RTOL = 1e-5
TORCH_F32_ATOL = 1e-6


@pytest.fixture(autouse=True)
def _deterministic_and_double():
    """Every test runs seeded, in float64, and restores global state after.

    float64 is the default *for tests* because it is the only precision in
    which finite differences and cross-framework comparisons are meaningful.
    The library's own default stays float32; tests that care assert on it
    explicitly.
    """
    previous = tt.get_default_dtype()
    tt.set_default_dtype(np.float64)
    tt.manual_seed(1234)
    yield
    tt.set_default_dtype(previous)


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh, independently seeded generator for data construction."""
    return np.random.default_rng(20260903)


def tensor64(array, requires_grad: bool = True) -> tt.Tensor:
    """Build a float64 TinyTorch tensor (the gradcheck-safe constructor)."""
    return tt.Tensor(np.asarray(array, dtype=np.float64), requires_grad=requires_grad)


def assert_allclose(actual, desired, rtol: float, atol: float, what: str = "value") -> None:
    """``np.testing.assert_allclose`` that accepts Tensors on either side."""
    a = actual.data if isinstance(actual, tt.Tensor) else np.asarray(actual)
    d = desired.data if isinstance(desired, tt.Tensor) else np.asarray(desired)
    np.testing.assert_allclose(
        np.asarray(a, dtype=np.float64),
        np.asarray(d, dtype=np.float64),
        rtol=rtol,
        atol=atol,
        err_msg=f"{what} mismatch",
    )
