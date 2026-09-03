"""Safe, deterministic checkpoint format.

Why not pickle
--------------
``pickle`` -- and therefore ``torch.save``'s default and ``np.save(...,
allow_pickle=True)`` -- deserialises by *executing* opcodes that can import
arbitrary modules and call arbitrary callables.  Loading an untrusted
checkpoint is equivalent to running untrusted code.  Model weights are
routinely downloaded from the internet, so that is a genuine attack surface,
not a theoretical one.

The ``.ttz`` format
-------------------
A ``.ttz`` file is a plain zip archive:

.. code-block:: text

    manifest.json          # format version + key order + dtype/shape per array
    arrays/<n>.npy         # one .npy per array, written with allow_pickle=False

Structure (nested dicts, lists, scalars, strings, bools, ``None``) lives in the
JSON manifest with array leaves replaced by ``{"__array__": n}`` references.
Loading reads the manifest, then the arrays -- **no code is executed**, and an
unknown or malformed manifest is rejected rather than interpreted.

Determinism
-----------
Saving the same state twice produces byte-identical files:

* dict keys are emitted in insertion order and the manifest records that order,
* every zip entry gets a fixed timestamp (the zip epoch, 1980-01-01) instead of
  the current time,
* compression is fixed to deflate level 6, and
* JSON is written with fixed separators and no ``sort_keys`` surprises.

That makes checkpoints hashable, diffable and cache-friendly, and lets a test
assert byte equality rather than approximate equality.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Any, BinaryIO

import numpy as np

from .tensor import Tensor

__all__ = ["FORMAT_VERSION", "SerializationError", "load", "save"]

FORMAT_VERSION = 1
_MANIFEST = "manifest.json"
_ARRAY_DIR = "arrays"
#: The zip epoch. Any fixed value works; this one is the minimum zip allows.
_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ALLOWED_SCALARS = (str, int, float, bool, type(None))


class SerializationError(Exception):
    """Raised for a malformed, truncated or unsupported checkpoint."""


def _encode(obj: Any, arrays: list[np.ndarray]) -> Any:
    """Replace array leaves with index references, recursively."""
    if isinstance(obj, Tensor):
        obj = obj.data
    if isinstance(obj, np.ndarray):
        arrays.append(np.ascontiguousarray(obj))
        return {"__array__": len(arrays) - 1}
    if isinstance(obj, np.generic):
        # 0-d NumPy scalars: keep them as arrays so dtype survives the trip.
        arrays.append(np.asarray(obj))
        return {"__array__": len(arrays) - 1}
    if isinstance(obj, dict):
        encoded = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise SerializationError(
                    f"dict keys must be strings for deterministic output, got {type(key).__name__}"
                )
            encoded[key] = _encode(value, arrays)
        return {"__dict__": encoded, "__order__": list(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return {"__list__": [_encode(v, arrays) for v in obj], "__tuple__": isinstance(obj, tuple)}
    if isinstance(obj, _ALLOWED_SCALARS):
        return {"__scalar__": obj}
    raise SerializationError(
        f"cannot serialize {type(obj).__name__}; supported types are Tensor, ndarray, "
        "dict, list, tuple, str, int, float, bool and None"
    )


def _decode(node: Any, arrays: list[np.ndarray]) -> Any:
    if not isinstance(node, dict):
        raise SerializationError("corrupt manifest: expected an object node")
    if "__array__" in node:
        index = node["__array__"]
        if not isinstance(index, int) or not 0 <= index < len(arrays):
            raise SerializationError(f"corrupt manifest: bad array reference {index!r}")
        return arrays[index]
    if "__dict__" in node:
        order = node.get("__order__", list(node["__dict__"].keys()))
        return {key: _decode(node["__dict__"][key], arrays) for key in order}
    if "__list__" in node:
        values = [_decode(v, arrays) for v in node["__list__"]]
        return tuple(values) if node.get("__tuple__") else values
    if "__scalar__" in node:
        value = node["__scalar__"]
        if not isinstance(value, _ALLOWED_SCALARS):
            raise SerializationError(f"corrupt manifest: bad scalar {type(value).__name__}")
        return value
    raise SerializationError(f"corrupt manifest: unrecognised node keys {sorted(node)}")


def _array_bytes(array: np.ndarray) -> bytes:
    """Serialise one array as a ``.npy`` payload with pickling disabled."""
    if array.dtype.hasobject:
        raise SerializationError(
            f"refusing to serialize object-dtype array (shape {array.shape}); "
            "that would require pickle"
        )
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def save(obj: Any, path: str | os.PathLike[str] | BinaryIO) -> None:
    """Write *obj* to a ``.ttz`` checkpoint.

    Parameters
    ----------
    obj:
        A nested structure of dicts/lists/tuples/scalars whose leaves are
        ``Tensor`` or ``ndarray`` -- typically a ``state_dict()``, or a dict
        bundling model state, optimizer state and metadata.
    path:
        Filesystem path or an open binary file object.

    Notes
    -----
    Output is byte-deterministic for equal input; see the module docstring.
    """
    arrays: list[np.ndarray] = []
    manifest = {
        "format": "tinytorch.ttz",
        "version": FORMAT_VERSION,
        "root": _encode(obj, arrays),
        "arrays": [{"dtype": str(a.dtype), "shape": list(a.shape)} for a in arrays],
    }
    payload = json.dumps(manifest, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def _write(stream: BinaryIO) -> None:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            info = zipfile.ZipInfo(_MANIFEST, date_time=_FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, payload)
            for index, array in enumerate(arrays):
                info = zipfile.ZipInfo(f"{_ARRAY_DIR}/{index}.npy", date_time=_FIXED_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, _array_bytes(array))

    if hasattr(path, "write"):
        _write(path)  # type: ignore[arg-type]
    else:
        with open(path, "wb") as handle:
            _write(handle)


def load(path: str | os.PathLike[str] | BinaryIO) -> Any:
    """Read a ``.ttz`` checkpoint written by :func:`save`.

    No code is executed: the manifest is parsed as JSON and arrays are read
    with ``allow_pickle=False``.
    """
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if _MANIFEST not in names:
                raise SerializationError("not a TinyTorch checkpoint: manifest.json is missing")
            manifest = json.loads(zf.read(_MANIFEST).decode("utf-8"))
            if manifest.get("format") != "tinytorch.ttz":
                raise SerializationError(f"unknown format tag {manifest.get('format')!r}")
            version = manifest.get("version")
            if version != FORMAT_VERSION:
                raise SerializationError(
                    f"checkpoint version {version} is not supported by this build "
                    f"(expected {FORMAT_VERSION})"
                )
            arrays: list[np.ndarray] = []
            for index, meta in enumerate(manifest.get("arrays", [])):
                entry = f"{_ARRAY_DIR}/{index}.npy"
                if entry not in names:
                    raise SerializationError(f"checkpoint is missing array {entry}")
                array = np.lib.format.read_array(io.BytesIO(zf.read(entry)), allow_pickle=False)
                if list(array.shape) != list(meta["shape"]) or str(array.dtype) != meta["dtype"]:
                    raise SerializationError(
                        f"array {index} does not match its manifest entry "
                        f"({array.dtype}{array.shape} vs {meta['dtype']}{tuple(meta['shape'])})"
                    )
                arrays.append(array)
            return _decode(manifest["root"], arrays)
    except zipfile.BadZipFile as exc:
        raise SerializationError("file is not a valid TinyTorch checkpoint (bad zip)") from exc
    except json.JSONDecodeError as exc:
        raise SerializationError("checkpoint manifest is not valid JSON") from exc
