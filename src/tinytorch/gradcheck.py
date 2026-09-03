"""Finite-difference gradient checking.

The single most valuable test for an autodiff engine is: *does the analytic
gradient agree with the definition of a derivative?*  This module answers that
by perturbing each input entry and measuring the change in the output.

Method
------
We use the **central** difference

.. math:: \\frac{\\partial f}{\\partial x_i} \\approx \\frac{f(x + h e_i) - f(x - h e_i)}{2h}

whose truncation error is :math:`O(h^2)`, versus :math:`O(h)` for the forward
difference -- worth the second function evaluation.

Precision is the real constraint.  Floating-point subtraction of two nearly
equal numbers loses about :math:`\\log_{10}(f/\\Delta f)` digits, so the total
error behaves like :math:`O(h^2) + O(\\epsilon/h)`.  With ``float64``
(:math:`\\epsilon \\approx 2.2\\times10^{-16}`) the optimum sits near
:math:`h \\approx 10^{-6}`, giving ~7 good digits -- hence the default
``eps=1e-6`` and ``rtol=1e-5``.  With ``float32`` (:math:`\\epsilon \\approx
1.2\\times10^{-7}`) there is *no* usable window, which is why
:func:`gradcheck` refuses to run on anything but ``float64``.

Comparison uses a **relative** criterion,
``|a - n| <= atol + rtol * |n|``, because gradient magnitudes across a network
span many orders of magnitude and a single absolute tolerance would be either
vacuous or impossible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from .autograd import no_grad
from .tensor import Tensor

__all__ = ["GradCheckError", "gradcheck", "numerical_gradient"]


class GradCheckError(AssertionError):
    """Raised when the analytic gradient disagrees with finite differences."""


def numerical_gradient(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    wrt: int = 0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central-difference gradient of ``fn(*inputs).sum()`` w.r.t. ``inputs[wrt]``.

    ``fn`` may return a non-scalar; it is summed, which is equivalent to seeding
    the backward pass with ``ones``.  Perturbation happens **in place** on the
    input's storage and is always restored, so the caller's tensors are
    unchanged on return.
    """
    target = inputs[wrt]
    grad = np.zeros(target.data.shape, dtype=np.float64)
    flat = target.data.reshape(-1)

    with no_grad():
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + eps
            plus = float(np.asarray(fn(*inputs).data, dtype=np.float64).sum())
            flat[i] = original - eps
            minus = float(np.asarray(fn(*inputs).data, dtype=np.float64).sum())
            flat[i] = original
            grad.reshape(-1)[i] = (plus - minus) / (2.0 * eps)
    return grad


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Iterable[Tensor],
    eps: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    raise_exception: bool = True,
) -> bool:
    """Verify every input's analytic gradient against finite differences.

    Parameters
    ----------
    fn:
        Callable taking the tensors in *inputs* and returning a Tensor.  It is
        summed to a scalar before differentiating.
    inputs:
        Tensors to check.  Those with ``requires_grad=False`` are skipped.
        **All must be ``float64``** -- see the module docstring.
    eps:
        Finite-difference step.  ``1e-6`` is near optimal for ``float64``.
    rtol, atol:
        Passed to ``|a - n| <= atol + rtol * |n|``.

    Returns
    -------
    bool
        ``True`` on success.  On failure raises :class:`GradCheckError` unless
        ``raise_exception=False``, in which case ``False`` is returned.
    """
    inputs = list(inputs)
    for i, t in enumerate(inputs):
        if t.requires_grad and t.data.dtype != np.float64:
            raise TypeError(
                f"gradcheck input {i} has dtype {t.data.dtype}; float64 is required "
                "because float32 cannot resolve a central difference "
                "(see tinytorch.gradcheck module docs)"
            )

    # Analytic pass: one backward, seeded with ones to match sum().
    for t in inputs:
        t.grad = None
    out = fn(*inputs)
    out.backward(np.ones_like(out.data))
    analytic = [None if t.grad is None else t.grad.copy() for t in inputs]

    failures: list[str] = []
    for i, t in enumerate(inputs):
        if not t.requires_grad:
            continue
        numeric = numerical_gradient(fn, inputs, wrt=i, eps=eps)
        got = analytic[i]
        if got is None:
            got = np.zeros_like(numeric)
        if got.shape != numeric.shape:
            failures.append(
                f"input {i}: analytic gradient shape {got.shape} != input shape {numeric.shape}"
            )
            continue
        diff = np.abs(got.astype(np.float64) - numeric)
        tol = atol + rtol * np.abs(numeric)
        bad = diff > tol
        if bad.any():
            worst = int(np.argmax(diff - tol))
            failures.append(
                f"input {i}: {int(bad.sum())}/{bad.size} entries exceed tolerance; "
                f"worst at flat index {worst}: analytic={got.reshape(-1)[worst]:.12g} "
                f"numeric={numeric.reshape(-1)[worst]:.12g} "
                f"|diff|={diff.reshape(-1)[worst]:.3e} tol={tol.reshape(-1)[worst]:.3e}"
            )

    if failures:
        if raise_exception:
            raise GradCheckError("gradcheck failed:\n  " + "\n  ".join(failures))
        return False
    return True
