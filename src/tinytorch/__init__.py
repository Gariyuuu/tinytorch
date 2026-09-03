"""TinyTorch -- a small, readable deep-learning framework built on NumPy.

TinyTorch implements reverse-mode automatic differentiation, a dynamic
computation graph, a ``Module``/``Parameter`` neural-network API, optimizers and
a safe serialization format from first principles.  NumPy is used *only* as the
numerical kernel: every derivative rule, the graph and the backward traversal
are TinyTorch's own.

    >>> import tinytorch as tt
    >>> from tinytorch import nn, optim
    >>> model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    >>> opt = optim.Adam(model.parameters(), lr=1e-2)
    >>> loss = nn.MSELoss()(model(tt.randn(16, 4)), tt.zeros(16, 1))
    >>> opt.zero_grad(); loss.backward(); opt.step()
"""

from __future__ import annotations

from . import nn, ops, optim
from .autograd import Context, Function, enable_grad, is_grad_enabled, no_grad, set_grad_enabled
from .creation import (
    arange,
    empty,
    eye,
    from_numpy,
    full,
    linspace,
    manual_seed,
    ones,
    ones_like,
    rand,
    randn,
    zeros,
    zeros_like,
)
from .gradcheck import gradcheck, numerical_gradient
from .ops import (
    add,
    div,
    exp,
    gelu,
    log,
    log_softmax,
    matmul,
    max,
    maximum,
    mean,
    minimum,
    mul,
    neg,
    pow,
    relu,
    reshape,
    sigmoid,
    softmax,
    sub,
    sum,
    tanh,
    transpose,
)
from .serialization import load, save
from .tensor import Tensor, get_default_dtype, set_default_dtype, tensor

__version__ = "0.1.0"

__all__ = [
    "Context",
    "Function",
    "Tensor",
    "__version__",
    "add",
    "arange",
    "div",
    "empty",
    "enable_grad",
    "exp",
    "eye",
    "from_numpy",
    "full",
    "gelu",
    "get_default_dtype",
    "gradcheck",
    "is_grad_enabled",
    "linspace",
    "load",
    "log",
    "log_softmax",
    "manual_seed",
    "matmul",
    "max",
    "maximum",
    "mean",
    "minimum",
    "mul",
    "neg",
    "nn",
    "no_grad",
    "numerical_gradient",
    "ones",
    "ones_like",
    "ops",
    "optim",
    "pow",
    "rand",
    "randn",
    "relu",
    "reshape",
    "save",
    "set_default_dtype",
    "set_grad_enabled",
    "sigmoid",
    "softmax",
    "sub",
    "sum",
    "tanh",
    "tensor",
    "transpose",
    "zeros",
    "zeros_like",
]
