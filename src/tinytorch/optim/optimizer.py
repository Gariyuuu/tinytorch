"""Optimizer base class: parameter groups and per-parameter state.

An optimizer owns two things.

**Parameter groups** let different parts of a model train under different
hyperparameters -- a lower learning rate for a pretrained trunk, no weight
decay on biases -- without needing two optimizers. Each group is a dict with a
``params`` list plus whatever defaults the subclass declares.

**Per-parameter state** is where momentum buffers, second-moment estimates and
step counters live. It is keyed by parameter *identity*, not name, because the
optimizer never sees the model's naming: it is handed a bare iterable of
tensors. That is also why :meth:`state_dict` re-keys state by the parameter's
positional index -- ``id()`` is meaningless once a process exits.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from ..tensor import Tensor

__all__ = ["Optimizer"]


class Optimizer:
    """Base class for optimization algorithms.

    Parameters
    ----------
    params:
        An iterable of :class:`~tinytorch.tensor.Tensor` (typically
        ``model.parameters()``), or an iterable of parameter-group dicts.
    defaults:
        Hyperparameter defaults applied to every group that omits them.
    """

    def __init__(self, params: Iterable[Any], defaults: dict[str, Any]) -> None:
        self.defaults = dict(defaults)
        self.state: dict[int, dict[str, Any]] = {}
        self.param_groups: list[dict[str, Any]] = []

        param_list = list(params)
        if not param_list:
            raise ValueError("optimizer got an empty parameter list")

        groups = param_list if isinstance(param_list[0], dict) else [{"params": param_list}]
        for group in groups:
            self.add_param_group(dict(group))

    def add_param_group(self, group: dict[str, Any]) -> None:
        """Register a parameter group, filling in missing hyperparameters."""
        params = group.get("params")
        if params is None:
            raise ValueError("parameter group must contain a 'params' key")
        if isinstance(params, Tensor):
            params = [params]
        params = list(params)
        for param in params:
            if not isinstance(param, Tensor):
                raise TypeError(f"optimizer can only optimize Tensors, got {type(param).__name__}")
            if not param.requires_grad:
                raise ValueError(
                    "optimizer got a parameter with requires_grad=False; it would never update"
                )
        seen: set[int] = {id(p) for g in self.param_groups for p in g["params"]}
        duplicates = [p for p in params if id(p) in seen]
        if duplicates:
            raise ValueError("a parameter appears in more than one parameter group")

        group["params"] = params
        for key, value in self.defaults.items():
            group.setdefault(key, value)
        self.param_groups.append(group)

    # -- iteration ----------------------------------------------------------
    def _params(self) -> Iterator[tuple[dict[str, Any], Tensor]]:
        for group in self.param_groups:
            for param in group["params"]:
                yield group, param

    def _flat_params(self) -> list[Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    def _get_state(self, param: Tensor) -> dict[str, Any]:
        """Per-parameter scratch dict, created lazily on first update.

        Lazy creation is what makes "step 1" detectable: an empty state dict
        means this parameter has never been updated, which both SGD-with-
        momentum and Adam need in order to initialise their buffers correctly.
        """
        key = id(param)
        if key not in self.state:
            self.state[key] = {}
        return self.state[key]

    # -- API ----------------------------------------------------------------
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients on every parameter this optimizer owns.

        Call this **before** ``loss.backward()`` on every iteration: backward
        accumulates, so skipping it silently trains on the sum of all gradients
        seen so far.
        """
        for _, param in self._params():
            param.zero_grad(set_to_none)

    def step(self) -> None:
        """Apply one update using the currently accumulated gradients."""
        raise NotImplementedError

    # -- checkpointing ------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Serializable snapshot: hyperparameters plus per-parameter state.

        State is re-keyed from ``id(param)`` to the parameter's position in the
        flattened group order, which is stable across processes as long as the
        model is constructed the same way.
        """
        index = {id(p): i for i, p in enumerate(self._flat_params())}
        packed_state: dict[str, Any] = {}
        for key, value in self.state.items():
            if key not in index:
                continue
            packed_state[str(index[key])] = {
                k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in value.items()
            }
        groups = []
        offset = 0
        for group in self.param_groups:
            entry = {k: v for k, v in group.items() if k != "params"}
            entry["params"] = list(range(offset, offset + len(group["params"])))
            offset += len(group["params"])
            groups.append(entry)
        return {"state": packed_state, "param_groups": groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore from :meth:`state_dict`, re-binding state to live parameters."""
        flat = self._flat_params()
        groups = state_dict.get("param_groups", [])
        if sum(len(g["params"]) for g in groups) != len(flat):
            raise ValueError("optimizer state_dict does not match the current parameter layout")

        self.state = {}
        for str_index, value in state_dict.get("state", {}).items():
            param = flat[int(str_index)]
            self.state[id(param)] = {
                k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                for k, v in value.items()
            }
        for group, saved in zip(self.param_groups, groups, strict=True):
            for key, value in saved.items():
                if key != "params":
                    group[key] = value

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for i, group in enumerate(self.param_groups):
            lines.append(f"  group {i}: {len(group['params'])} tensors")
            for key in sorted(k for k in group if k != "params"):
                lines.append(f"    {key}: {group[key]}")
        lines.append(")")
        return "\n".join(lines)
