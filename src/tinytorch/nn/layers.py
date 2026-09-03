"""Layers with learnable parameters, plus containers."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

from .. import ops
from ..creation import zeros
from ..tensor import Tensor
from . import init
from .module import Module, Parameter

__all__ = ["Dropout", "Flatten", "Identity", "Linear", "Sequential"]


class Linear(Module):
    r"""Affine map :math:`y = xW^\top + b`.

    The weight is stored ``(out_features, in_features)`` -- the *transposed*
    layout -- and the forward pass transposes it.  That is PyTorch's
    convention and it exists so a row of ``W`` is one output unit's weight
    vector, which is the contiguous layout row-major memory wants.

    Gradients follow directly from :class:`~tinytorch.ops.MatMul`:
    :math:`\bar W = \bar y^\top x` and :math:`\bar b = \sum_{\text{batch}} \bar y`
    (the bias is broadcast over the batch, so its gradient is the sum over it --
    handled automatically by the engine's unbroadcast step).

    Default initialisation matches ``torch.nn.Linear``: Kaiming-uniform with
    ``a=sqrt(5)`` for the weight (which works out to bound
    ``1/sqrt(fan_in)``), and the same bound for the bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(zeros(self.out_features, self.in_features).data)
        if bias:
            self.bias = Parameter(zeros(self.out_features).data)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """(Re)initialise weight and bias in place."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init.calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out = ops.matmul(x, ops.transpose(self.weight))
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class Sequential(Module):
    """Chain modules; the output of each is the input to the next.

    Children are registered under their positional index, so ``state_dict``
    keys read ``"0.weight"``, ``"1.weight"`` -- the same as PyTorch, which
    keeps checkpoints comparable between the two.

    >>> Sequential(Linear(2, 8), ReLU(), Linear(8, 1))
    """

    def __init__(self, *modules: Any) -> None:
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], (OrderedDict, dict)):
            for name, module in modules[0].items():
                self.add_module(str(name), module)
        else:
            for index, module in enumerate(modules):
                self.add_module(str(index), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            if module is not None:
                x = module(x)
        return x

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(m for m in self._modules.values() if m is not None)

    def __getitem__(self, index: int | slice) -> Module | Sequential:
        items = list(self._modules.values())
        if isinstance(index, slice):
            return Sequential(*items[index])
        return items[index]

    def append(self, module: Module) -> Sequential:
        """Add a module at the end."""
        self.add_module(str(len(self._modules)), module)
        return self


class Identity(Module):
    """Returns its input unchanged (useful as a configurable no-op)."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class Flatten(Module):
    """Collapse all dimensions from *start_dim* onward into one."""

    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}"


class Dropout(Module):
    """Zero each element with probability *p* during training only.

    Uses **inverted** dropout: surviving activations are scaled by
    ``1/(1-p)`` at training time, so the expected value of the layer's output
    is unchanged and evaluation needs no rescaling at all -- ``eval()`` simply
    returns the input.  This is the one place ``self.training`` changes the
    math, which is why :meth:`Module.train` / :meth:`Module.eval` exist.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        from ..creation import default_generator

        keep = default_generator().random(x.shape) >= self.p
        mask = Tensor((keep / (1.0 - self.p)).astype(x.dtype), _copy=False)
        return ops.mul(x, mask)

    def extra_repr(self) -> str:
        return f"p={self.p}"
