"""Checkpoint format: correctness, determinism, safety and error handling."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

import tinytorch as tt
from tinytorch import nn, optim
from tinytorch.serialization import SerializationError


# --------------------------------------------------------------------------
# Round-tripping structures.
# --------------------------------------------------------------------------
def test_round_trip_arrays_and_scalars(tmp_path):
    original = {
        "weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "step": 7,
        "lr": 1e-3,
        "name": "layer1",
        "flag": True,
        "nothing": None,
        "betas": (0.9, 0.999),
        "history": [1.0, 2.0, 3.0],
        "nested": {"inner": np.ones((2, 2), dtype=np.float64)},
    }
    path = tmp_path / "ckpt.ttz"
    tt.save(original, path)
    loaded = tt.load(path)

    np.testing.assert_array_equal(loaded["weight"], original["weight"])
    assert loaded["weight"].dtype == np.float32
    assert loaded["step"] == 7 and isinstance(loaded["step"], int)
    assert loaded["lr"] == 1e-3
    assert loaded["name"] == "layer1"
    assert loaded["flag"] is True
    assert loaded["nothing"] is None
    assert loaded["betas"] == (0.9, 0.999)
    assert loaded["history"] == [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(loaded["nested"]["inner"], original["nested"]["inner"])


def test_tensors_are_saved_as_arrays(tmp_path):
    path = tmp_path / "t.ttz"
    tt.save({"x": tt.Tensor([1.0, 2.0], requires_grad=True)}, path)
    loaded = tt.load(path)
    assert isinstance(loaded["x"], np.ndarray)
    np.testing.assert_allclose(loaded["x"], [1.0, 2.0])


def test_dict_key_order_is_preserved(tmp_path):
    original = {"z": np.zeros(1), "a": np.ones(1), "m": np.full(1, 2.0)}
    path = tmp_path / "order.ttz"
    tt.save(original, path)
    assert list(tt.load(path)) == ["z", "a", "m"]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.int64, np.bool_])
def test_dtypes_survive_the_round_trip(tmp_path, dtype):
    array = np.ones((2, 3), dtype=dtype)
    path = tmp_path / "dtype.ttz"
    tt.save({"a": array}, path)
    loaded = tt.load(path)["a"]
    assert loaded.dtype == dtype
    np.testing.assert_array_equal(loaded, array)


def test_save_to_a_file_object(tmp_path):
    buffer = io.BytesIO()
    tt.save({"a": np.arange(4.0)}, buffer)
    buffer.seek(0)
    np.testing.assert_allclose(tt.load(buffer)["a"], np.arange(4.0))


def test_non_contiguous_arrays_are_handled(tmp_path):
    array = np.arange(24.0).reshape(4, 6)[:, ::2]
    assert not array.flags["C_CONTIGUOUS"]
    path = tmp_path / "strided.ttz"
    tt.save({"a": array}, path)
    np.testing.assert_array_equal(tt.load(path)["a"], array)


# --------------------------------------------------------------------------
# Determinism.
# --------------------------------------------------------------------------
def test_saving_the_same_state_twice_is_byte_identical(tmp_path):
    """Fixed zip timestamps + fixed key order => reproducible checkpoints."""
    state = {"w": np.arange(20.0).reshape(4, 5), "step": 3}
    first, second = tmp_path / "a.ttz", tmp_path / "b.ttz"
    tt.save(state, first)
    tt.save(state, second)
    assert first.read_bytes() == second.read_bytes()


def test_zip_entries_have_a_fixed_timestamp(tmp_path):
    path = tmp_path / "stamp.ttz"
    tt.save({"a": np.ones(3)}, path)
    with zipfile.ZipFile(path) as zf:
        assert {info.date_time for info in zf.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_a_changed_value_changes_the_bytes(tmp_path):
    """Determinism must not be achieved by ignoring the content."""
    a, b = tmp_path / "a.ttz", tmp_path / "b.ttz"
    tt.save({"w": np.ones(4)}, a)
    tt.save({"w": np.full(4, 1.0000001)}, b)
    assert a.read_bytes() != b.read_bytes()


# --------------------------------------------------------------------------
# Safety.
# --------------------------------------------------------------------------
def test_the_archive_contains_only_json_and_npy(tmp_path):
    """Nothing in the file can name a Python module or callable."""
    path = tmp_path / "safe.ttz"
    tt.save({"w": np.ones(3), "meta": {"note": "hello"}}, path)
    with zipfile.ZipFile(path) as zf:
        names = sorted(zf.namelist())
        assert names == ["arrays/0.npy", "manifest.json"]
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["format"] == "tinytorch.ttz"
        assert manifest["version"] == 1


def test_object_dtype_arrays_are_refused(tmp_path):
    """Object arrays can only be stored via pickle, so we refuse them."""
    with pytest.raises(SerializationError, match="object-dtype"):
        tt.save({"bad": np.array([{"a": 1}], dtype=object)}, tmp_path / "x.ttz")


def test_unsupported_python_types_are_refused(tmp_path):
    with pytest.raises(SerializationError, match="cannot serialize"):
        tt.save({"fn": len}, tmp_path / "x.ttz")


def test_non_string_dict_keys_are_refused(tmp_path):
    with pytest.raises(SerializationError, match="dict keys must be strings"):
        tt.save({1: np.ones(2)}, tmp_path / "x.ttz")


# --------------------------------------------------------------------------
# Error handling on load.
# --------------------------------------------------------------------------
def test_loading_a_non_zip_file_fails_cleanly(tmp_path):
    path = tmp_path / "junk.ttz"
    path.write_bytes(b"this is not a zip archive")
    with pytest.raises(SerializationError, match="bad zip"):
        tt.load(path)


def test_missing_manifest_fails_cleanly(tmp_path):
    path = tmp_path / "empty.ttz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("something.txt", "hi")
    with pytest.raises(SerializationError, match=r"manifest\.json is missing"):
        tt.load(path)


def test_wrong_format_tag_is_rejected(tmp_path):
    path = tmp_path / "wrong.ttz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": "something-else"}))
    with pytest.raises(SerializationError, match="unknown format tag"):
        tt.load(path)


def test_future_version_is_rejected(tmp_path):
    path = tmp_path / "v99.ttz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"format": "tinytorch.ttz", "version": 99, "root": {}, "arrays": []}),
        )
    with pytest.raises(SerializationError, match="version 99 is not supported"):
        tt.load(path)


def test_missing_array_entry_is_detected(tmp_path):
    path = tmp_path / "truncated.ttz"
    tt.save({"w": np.ones(3)}, path)
    data = zipfile.ZipFile(path).read("manifest.json")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", data)  # arrays/0.npy deliberately dropped
    with pytest.raises(SerializationError, match="missing array"):
        tt.load(path)


def test_corrupt_manifest_json_is_detected(tmp_path):
    path = tmp_path / "corrupt.ttz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", "{not json")
    with pytest.raises(SerializationError, match="not valid JSON"):
        tt.load(path)


def test_bad_array_reference_is_detected(tmp_path):
    path = tmp_path / "badref.ttz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "format": "tinytorch.ttz",
                    "version": 1,
                    "root": {"__array__": 5},
                    "arrays": [],
                }
            ),
        )
    with pytest.raises(SerializationError, match="bad array reference"):
        tt.load(path)


# --------------------------------------------------------------------------
# End-to-end model and optimizer checkpointing.
# --------------------------------------------------------------------------
def test_model_checkpoint_reproduces_predictions(tmp_path):
    tt.manual_seed(11)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    x = tt.Tensor(np.random.default_rng(12).standard_normal((6, 4)))
    expected = model(x).data.copy()

    path = tmp_path / "model.ttz"
    tt.save(model.state_dict(), path)

    tt.manual_seed(999)  # different init on purpose
    restored = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    assert not np.allclose(restored(x).data, expected)
    restored.load_state_dict(tt.load(path))
    np.testing.assert_array_equal(restored(x).data, expected)


def test_full_training_checkpoint_resumes_exactly(tmp_path):
    """Model + optimizer + epoch counter through the real save/load path."""
    def build():
        tt.manual_seed(13)
        model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 1))
        return model, optim.Adam(model.parameters(), lr=0.02)

    x = tt.Tensor(np.random.default_rng(14).standard_normal((10, 3)))
    y = tt.Tensor(np.random.default_rng(15).standard_normal((10, 1)))

    def train(model, opt, steps):
        for _ in range(steps):
            opt.zero_grad()
            nn.MSELoss()(model(x), y).backward()
            opt.step()

    reference, ref_opt = build()
    train(reference, ref_opt, 8)

    partial, partial_opt = build()
    train(partial, partial_opt, 4)
    path = tmp_path / "full.ttz"
    tt.save(
        {"model": partial.state_dict(), "optim": partial_opt.state_dict(), "epoch": 4}, path
    )

    checkpoint = tt.load(path)
    assert checkpoint["epoch"] == 4
    resumed, resumed_opt = build()
    resumed.load_state_dict(checkpoint["model"])
    resumed_opt.load_state_dict(checkpoint["optim"])
    train(resumed, resumed_opt, 4)

    for (name, a), (_, b) in zip(reference.named_parameters(), resumed.named_parameters(), strict=True):
        np.testing.assert_allclose(b.data, a.data, rtol=1e-12, err_msg=name)


def test_checkpoint_of_an_identical_model_is_byte_identical(tmp_path):
    """Two models trained identically produce identical checkpoint files."""
    def train_one():
        tt.manual_seed(16)
        model = nn.Linear(4, 2)
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        x = tt.Tensor(np.ones((5, 4)))
        for _ in range(3):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()
        return model

    a, b = tmp_path / "a.ttz", tmp_path / "b.ttz"
    tt.save(train_one().state_dict(), a)
    tt.save(train_one().state_dict(), b)
    assert a.read_bytes() == b.read_bytes()
