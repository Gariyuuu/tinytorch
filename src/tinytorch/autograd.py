"""Reverse-mode automatic differentiation engine.

This module owns everything about *differentiation*.  NumPy is used purely as
the numerical kernel (storage + elementwise/BLAS math); the computation graph,
the chain rule and the backward traversal are implemented here from scratch.

Design
------
TinyTorch builds a **dynamic** (define-by-run) graph.  There is no compilation
step and no static graph object: every differentiable operation is a
:class:`Function` whose :meth:`Function.apply` classmethod

1. unwraps its :class:`~tinytorch.tensor.Tensor` inputs into ``np.ndarray``,
2. runs ``forward`` to get the output storage,
3. wraps the result in a new ``Tensor``, and
4. if any input requires gradient *and* grad mode is enabled, records a
   :class:`Node` on the output describing how to invert step 2.

The resulting structure is a DAG whose edges point from an output back to the
inputs that produced it (``Tensor.grad_fn.inputs``).  ``backward`` walks that
DAG in reverse topological order, multiplying local Jacobian-vector products
together -- the chain rule -- and accumulating the result into ``Tensor.grad``
for every leaf.

Two invariants keep the engine small and correct:

* **Gradients are plain ``np.ndarray``**, never ``Tensor``.  This makes the
  engine non-differentiable (no double backward) but removes any chance of the
  backward pass accidentally extending the forward graph.
* **A ``Tensor`` has at most one ``grad_fn``.**  Reuse of a tensor by several
  consumers is therefore handled entirely by accumulation in the traversal, not
  by graph surgery.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tensor import Tensor

__all__ = [
    "Context",
    "Function",
    "Node",
    "backward",
    "enable_grad",
    "is_grad_enabled",
    "no_grad",
    "set_grad_enabled",
    "unbroadcast",
]

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Grad mode
# ---------------------------------------------------------------------------
#
# Grad mode is thread-local: two threads running inference and training
# concurrently must not clobber each other's setting.
class _GradMode(threading.local):
    enabled: bool = True


_grad_mode = _GradMode()


def is_grad_enabled() -> bool:
    """Return whether new operations will record autograd history."""
    return _grad_mode.enabled


class set_grad_enabled:
    """Context manager / decorator that sets grad mode to *mode*.

    >>> with set_grad_enabled(False):
    ...     y = model(x)          # no graph is recorded
    """

    def __init__(self, mode: bool) -> None:
        self.mode = bool(mode)
        self.prev = is_grad_enabled()

    def __enter__(self) -> set_grad_enabled:
        self.prev = is_grad_enabled()
        _grad_mode.enabled = self.mode
        return self

    def __exit__(self, *exc: object) -> None:
        _grad_mode.enabled = self.prev

    def __call__(self, fn: F) -> F:
        import functools

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with set_grad_enabled(self.mode):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


class no_grad(set_grad_enabled):
    """Disable graph recording inside the block (inference / optimizer math)."""

    def __init__(self) -> None:
        super().__init__(False)


class enable_grad(set_grad_enabled):
    """Re-enable graph recording inside a :class:`no_grad` block."""

    def __init__(self) -> None:
        super().__init__(True)


# ---------------------------------------------------------------------------
# Broadcasting-aware gradient reduction
# ---------------------------------------------------------------------------
def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce *grad* so that it has ``shape``, undoing NumPy broadcasting.

    Broadcasting is a *copy* operation: ``x`` of shape ``(1, 3)`` used against
    ``(4, 3)`` is implicitly tiled four times.  A copy is a fan-out in the
    graph, and the adjoint of a fan-out is a **sum**.  So the vector-Jacobian
    product for a broadcast input is obtained by summing the incoming gradient
    over every axis that was expanded:

    * axes NumPy *prepended* (``grad.ndim > len(shape)``) are summed away, and
    * axes where the input had extent 1 but the output did not are summed with
      ``keepdims=True`` so the singleton axis survives.

    Doing this in one place means every broadcasting op gets correct gradients
    for free.
    """
    if grad.shape == shape:
        return grad

    # 1. Collapse the leading axes NumPy invented.
    extra_dims = grad.ndim - len(shape)
    if extra_dims > 0:
        grad = grad.sum(axis=tuple(range(extra_dims)))

    # 2. Collapse axes that were stretched from extent 1.
    stretched = tuple(i for i, dim in enumerate(shape) if dim == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)

    # Guard against a shape mismatch that broadcasting could not explain.
    if grad.shape != shape:
        grad = grad.reshape(shape)
    return grad


# ---------------------------------------------------------------------------
# Function abstraction
# ---------------------------------------------------------------------------
class Context:
    """Scratch space handed to ``forward`` and read back by ``backward``.

    ``forward`` runs eagerly and then is gone; anything ``backward`` needs --
    input arrays, output arrays, shapes, reduction axes -- must be stashed
    here.  Keeping it in an explicit object (rather than on the Function
    instance) makes the "a Function is a pure pair of maps" story honest and
    mirrors the way PyTorch's ``autograd.Function`` works.
    """

    __slots__ = ("__dict__", "saved_tensors")

    #: Arrays saved by ``save_for_backward``.
    saved_tensors: tuple[np.ndarray, ...]

    # Ops also stash plain metadata straight onto the context. These are
    # annotation-only declarations: they cost nothing at runtime (no class
    # attribute is created) but they document the vocabulary the built-in ops
    # use and let a type checker see the assignments in forward().
    shape: tuple[int, ...]  #: input shape, for ops that reduce or reshape
    dtype: Any  #: input dtype, for ops that allocate the gradient themselves
    axis: Any  #: reduced/normalised axis or axes
    axes: Any  #: permutation for Transpose
    keepdims: bool  #: whether a reduction kept its collapsed axes
    count: int  #: number of elements averaged, for Mean
    index: Any  #: index expression for GetItem
    approximate: str  #: GELU variant

    def __getattr__(self, name: str) -> Any:
        # Only reached when an op forgot to save something in forward().
        raise AttributeError(f"Context has no attribute {name!r}; was it set in forward()?")

    def __init__(self) -> None:
        self.saved_tensors = ()

    def save_for_backward(self, *arrays: np.ndarray) -> None:
        """Stash arrays needed to evaluate the local derivative."""
        self.saved_tensors = arrays


class Function:
    """Base class for a differentiable primitive.

    Subclasses implement two static maps over **raw NumPy arrays**:

    ``forward(ctx, *inputs, **params) -> np.ndarray``
        The primal computation.

    ``backward(ctx, grad_output) -> np.ndarray | tuple[np.ndarray | None, ...]``
        The vector-Jacobian product: given :math:`\\bar y = \\partial L/\\partial y`,
        return :math:`\\bar x_i = \\partial L/\\partial x_i` for each *positional*
        input, in order.  Return ``None`` in a slot whose input is not
        differentiable.

    Non-differentiable configuration (axes, exponents used as constants,
    indices, ...) is passed as **keyword** arguments, which is what makes the
    "one returned gradient per positional argument" contract unambiguous.
    """

    #: Human readable name used when printing the graph.
    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @staticmethod
    def forward(ctx: Context, *args: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(
        ctx: Context, grad_output: np.ndarray
    ) -> np.ndarray | tuple[np.ndarray | None, ...] | None:
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Tensor:
        """Run the op and, when required, splice a node into the graph."""
        from .tensor import Tensor  # local import: tensor.py imports ops.py

        raw = tuple(a.data if isinstance(a, Tensor) else a for a in args)
        ctx = Context()
        out_data = cls.forward(ctx, *raw, **kwargs)
        out_data = np.asarray(out_data)

        requires_grad = is_grad_enabled() and any(
            isinstance(a, Tensor) and a.requires_grad for a in args
        )
        out = Tensor(out_data, requires_grad=requires_grad, _copy=False)
        if requires_grad:
            out.grad_fn = Node(cls, ctx, args)
        return out


class Node:
    """A recorded operation: the edge set of the backward graph.

    ``inputs`` keeps the *original* positional arguments (tensors and plain
    Python scalars alike) so that the tuple ``backward`` returns can be zipped
    against them positionally.
    """

    __slots__ = ("ctx", "fn", "inputs")

    def __init__(self, fn: type[Function], ctx: Context, inputs: tuple[Any, ...]) -> None:
        self.fn = fn
        self.ctx = ctx
        self.inputs = inputs

    @property
    def name(self) -> str:
        return self.fn.name()

    def apply_backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        """Evaluate the local VJP, normalised to one entry per input."""
        grads = self.fn.backward(self.ctx, grad_output)
        if not isinstance(grads, tuple):
            grads = (grads,)
        if len(grads) != len(self.inputs):
            raise RuntimeError(
                f"{self.name}.backward returned {len(grads)} gradient(s) but "
                f"forward received {len(self.inputs)} positional input(s)"
            )
        return grads

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Node {self.name}>"


# ---------------------------------------------------------------------------
# Backward traversal
# ---------------------------------------------------------------------------
def _topological_order(root: Tensor) -> list[Tensor]:
    """Post-order DFS over the backward DAG, reachable from *root*.

    Returned in *dependency-first* order (inputs before the outputs that
    consume them); the engine then walks it in reverse so a tensor's gradient
    is complete before it is propagated further.  The traversal is iterative
    rather than recursive: a 500-layer network would otherwise exhaust
    CPython's recursion limit.
    """
    order: list[Tensor] = []
    visited: set[int] = set()
    # Each stack entry is (tensor, children_already_pushed).
    stack: list[tuple[Tensor, bool]] = [(root, False)]

    while stack:
        tensor, expanded = stack.pop()
        if expanded:
            order.append(tensor)
            continue
        if id(tensor) in visited:
            continue
        visited.add(id(tensor))
        stack.append((tensor, True))
        node = tensor.grad_fn
        if node is None:
            continue
        for inp in node.inputs:
            if _is_differentiable_tensor(inp) and id(inp) not in visited:
                stack.append((inp, False))
    return order


def _is_differentiable_tensor(obj: Any) -> bool:
    from .tensor import Tensor

    return isinstance(obj, Tensor) and obj.requires_grad


def backward(root: Tensor, gradient: np.ndarray | None = None) -> None:
    """Accumulate ``d(root)/d(leaf)`` into ``leaf.grad`` for every reachable leaf.

    Parameters
    ----------
    root:
        Tensor to differentiate.  Must be a scalar unless *gradient* is given.
    gradient:
        Seed :math:`\\bar y`.  Defaults to ``ones_like(root)`` for a scalar
        root, which is what makes ``loss.backward()`` mean "differentiate the
        loss".  For a non-scalar root this is the vector the Jacobian is
        multiplied by -- TinyTorch computes vector-Jacobian products, never a
        full Jacobian.

    Notes
    -----
    Gradients **accumulate**: calling ``backward`` twice without an intervening
    ``zero_grad`` sums the two results.  That is deliberate -- it is what makes
    gradient accumulation over micro-batches and multi-loss training work --
    and it is why every training loop must call ``optimizer.zero_grad()``.
    """
    if not root.requires_grad:
        raise RuntimeError(
            "backward() called on a tensor that does not require grad; "
            "no graph was recorded for it"
        )

    if gradient is None:
        if root.data.size != 1:
            raise RuntimeError(
                "grad can be implicitly created only for scalar outputs; "
                f"root has shape {root.shape}. Pass an explicit `gradient=`."
            )
        seed = np.ones_like(root.data)
    else:
        seed = np.asarray(gradient)
        if seed.shape != root.data.shape:
            raise RuntimeError(
                f"gradient shape {seed.shape} does not match tensor shape {root.data.shape}"
            )

    order = _topological_order(root)
    # Pending gradient per tensor, keyed by identity (Tensors are unhashable by
    # value because __eq__ is an elementwise op).
    pending: dict[int, np.ndarray] = {id(root): seed}

    for tensor in reversed(order):
        grad = pending.pop(id(tensor), None)
        if grad is None:
            continue  # unreachable branch of the DAG

        node = tensor.grad_fn
        if node is None or tensor.retains_grad:
            # Leaf (or explicitly retained): this is where gradient lands.
            tensor._accumulate_grad(grad)
        if node is None:
            continue

        input_grads = node.apply_backward(grad)
        # Lengths are checked in apply_backward, so strict= is a free assertion.
        for inp, g in zip(node.inputs, input_grads, strict=True):
            if g is None or not _is_differentiable_tensor(inp):
                continue
            g = unbroadcast(np.asarray(g), inp.data.shape)
            key = id(inp)
            if key in pending:
                pending[key] = pending[key] + g
            else:
                pending[key] = g


def topological_order(root: Tensor) -> Sequence[Tensor]:
    """Public wrapper around the traversal, for tests and visualisation."""
    return _topological_order(root)
