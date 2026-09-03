"""``Module`` and ``Parameter``: the neural-network object model.

The central trick is attribute interception.  ``Module.__setattr__`` notices
when you assign a :class:`Parameter` or a nested :class:`Module` and files it
in ``_parameters`` / ``_modules``.  That is the whole of "parameter discovery":
writing ``self.weight = Parameter(...)`` in ``__init__`` is *simultaneously*
declaring the attribute and registering it for ``parameters()``,
``state_dict()`` and the optimizer.  No metaclass, no registry, no decorator.

Traversal is then a depth-first walk over ``_modules``, and every public API --
``parameters``, ``named_parameters``, ``state_dict``, ``train``, ``eval`` --
is a thin layer over that walk.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..tensor import Tensor

__all__ = ["Module", "Parameter"]


class Parameter(Tensor):
    """A :class:`~tinytorch.tensor.Tensor` that is a *learnable* model weight.

    Behaviourally it is just a tensor with ``requires_grad=True`` by default;
    the type is what makes it discoverable by :meth:`Module.parameters`.  That
    distinction matters: a constant buffer and a trainable weight are both
    tensors on a module, and only the type separates them.
    """

    def __init__(self, data: Any, requires_grad: bool = True) -> None:
        super().__init__(data, requires_grad=requires_grad)

    def __repr__(self) -> str:
        return "Parameter containing:\n" + super().__repr__()


class Module:
    """Base class for everything with state: layers, losses, whole models.

    Subclasses implement :meth:`forward`; calling the module invokes it.
    ``training`` is a flag every submodule inherits via :meth:`train` /
    :meth:`eval` -- layers whose behaviour differs between fitting and
    inference read it.
    """

    training: bool

    def __init__(self) -> None:
        # Bypass our own __setattr__, which needs these dicts to already exist.
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "training", True)

    # -- forward ------------------------------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    # -- registration -------------------------------------------------------
    def register_parameter(self, name: str, param: Parameter | None) -> None:
        """Explicitly register a parameter (usually ``self.x = Parameter(...)``)."""
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(f"{name} must be a Parameter, got {type(param).__name__}")
        self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Tensor | None) -> None:
        """Register non-learnable state that still belongs in ``state_dict``.

        Buffers travel with the model (they are saved and loaded) but are never
        handed to an optimizer -- running statistics, masks, lookup tables.
        """
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a Tensor, got {type(tensor).__name__}")
        self._buffers[name] = tensor

    def add_module(self, name: str, module: Module | None) -> None:
        """Register a child module under *name*."""
        if module is not None and not isinstance(module, Module):
            raise TypeError(f"{name} must be a Module, got {type(module).__name__}")
        self._modules[name] = module

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._pop_everywhere(name)
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._pop_everywhere(name)
            self._modules[name] = value
        elif name in self._parameters and value is None:
            self._parameters[name] = None
        elif name in self._buffers and (value is None or isinstance(value, Tensor)):
            self._buffers[name] = value
        else:
            object.__setattr__(self, name, value)

    def _pop_everywhere(self, name: str) -> None:
        object.__getattribute__(self, "__dict__").pop(name, None)
        self._parameters.pop(name, None)
        self._buffers.pop(name, None)
        self._modules.pop(name, None)

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails, i.e. for things we
        # diverted into the registries.
        for registry in ("_parameters", "_buffers", "_modules"):
            store = self.__dict__.get(registry)
            if store is not None and name in store:
                return store[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __delattr__(self, name: str) -> None:
        for registry in (self._parameters, self._buffers, self._modules):
            if name in registry:
                del registry[name]
                return
        object.__delattr__(self, name)

    # -- traversal ----------------------------------------------------------
    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        """Yield ``(qualified_name, module)`` for self and every descendant."""
        yield prefix, self
        for name, child in self._modules.items():
            if child is None:
                continue
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(child_prefix)

    def modules(self) -> Iterator[Module]:
        """Yield self and every descendant module."""
        for _, module in self.named_modules():
            yield module

    def children(self) -> Iterator[Module]:
        """Yield immediate child modules."""
        for child in self._modules.values():
            if child is not None:
                yield child

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        """Yield ``(qualified_name, parameter)``, deduplicated by identity.

        Deduplication matters for **weight tying**: if the same ``Parameter``
        object is reachable by two names, an optimizer must still see it once,
        or its update would be applied twice per step.
        """
        seen: set[int] = set()
        modules = self.named_modules(prefix) if recurse else [(prefix, self)]
        for module_name, module in modules:
            for name, param in module._parameters.items():
                if param is None or id(param) in seen:
                    continue
                seen.add(id(param))
                yield (f"{module_name}.{name}" if module_name else name), param

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Yield every learnable parameter, deduplicated."""
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[tuple[str, Tensor]]:
        """Yield ``(qualified_name, buffer)``."""
        seen: set[int] = set()
        modules = self.named_modules(prefix) if recurse else [(prefix, self)]
        for module_name, module in modules:
            for name, buf in module._buffers.items():
                if buf is None or id(buf) in seen:
                    continue
                seen.add(id(buf))
                yield (f"{module_name}.{name}" if module_name else name), buf

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Yield every registered buffer."""
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Total element count over parameters."""
        return sum(p.size for p in self.parameters() if p.requires_grad or not trainable_only)

    # -- modes --------------------------------------------------------------
    def train(self, mode: bool = True) -> Module:
        """Put this module and all descendants into training mode."""
        for module in self.modules():
            object.__setattr__(module, "training", bool(mode))
        return self

    def eval(self) -> Module:
        """Put this module and all descendants into evaluation mode."""
        return self.train(False)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear ``.grad`` on every parameter.

        Necessary because backward *accumulates*; see
        :func:`tinytorch.autograd.backward`.
        """
        for param in self.parameters():
            param.zero_grad(set_to_none)

    def apply(self, fn: Any) -> Module:
        """Apply *fn* to every submodule (typical use: custom initialisation)."""
        for module in self.modules():
            fn(module)
        return self

    # -- state --------------------------------------------------------------
    def state_dict(self) -> OrderedDict[str, np.ndarray]:
        """Flat ``{qualified_name: ndarray}`` snapshot of parameters and buffers.

        Arrays are **copies**, so the dict is a true snapshot: continuing to
        train will not mutate it.  Keys are ordered by declaration, which is
        what makes serialization deterministic.
        """
        state: OrderedDict[str, np.ndarray] = OrderedDict()
        for name, param in self.named_parameters():
            state[name] = param.data.copy()
        for name, buf in self.named_buffers():
            state[name] = buf.data.copy()
        return state

    def load_state_dict(
        self, state: dict[str, Any], strict: bool = True
    ) -> tuple[list[str], list[str]]:
        """Copy arrays from *state* into this module's tensors, in place.

        In place is deliberate: any optimizer already holding references to
        these parameters keeps working, and its momentum buffers stay valid.

        Returns ``(missing_keys, unexpected_keys)``; with ``strict=True`` either
        being non-empty is an error.
        """
        own: OrderedDict[str, Tensor] = OrderedDict()
        own.update(self.named_parameters())
        own.update(self.named_buffers())

        missing = [k for k in own if k not in state]
        unexpected = [k for k in state if k not in own]
        if strict and (missing or unexpected):
            raise KeyError(
                f"load_state_dict mismatch: missing={missing}, unexpected={unexpected}"
            )

        for key, tensor in own.items():
            if key not in state:
                continue
            value = np.asarray(state[key].data if isinstance(state[key], Tensor) else state[key])
            if value.shape != tensor.data.shape:
                raise ValueError(
                    f"size mismatch for {key}: checkpoint has {value.shape}, "
                    f"model expects {tensor.data.shape}"
                )
            tensor.copy_(value)
        return missing, unexpected

    # -- repr ---------------------------------------------------------------
    def extra_repr(self) -> str:
        """Override to describe a layer's configuration in ``repr``."""
        return ""

    def __repr__(self) -> str:
        lines = []
        extra = self.extra_repr()
        for name, child in self._modules.items():
            child_repr = repr(child).split("\n")
            head = f"  ({name}): {child_repr[0]}"
            tail = ["  " + line for line in child_repr[1:]]
            lines.extend([head, *tail])
        head = f"{type(self).__name__}("
        if not lines:
            return head + extra + ")"
        body = ([f"  {extra}"] if extra else []) + lines
        return "\n".join([head, *body, ")"])
