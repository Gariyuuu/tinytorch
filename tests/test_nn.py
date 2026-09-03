"""Module/Parameter discovery, mode switching, state dict, layers and init."""

from __future__ import annotations

import numpy as np
import pytest

import tinytorch as tt
from tinytorch import nn
from tinytorch.nn import init

from .conftest import tensor64


# --------------------------------------------------------------------------
# Parameter discovery.
# --------------------------------------------------------------------------
def test_assignment_registers_parameters_and_children():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(np.ones((2, 3)))
            self.inner = nn.Linear(3, 4)
            self.scalar = 7  # plain attributes must stay plain

    net = Net()
    names = [n for n, _ in net.named_parameters()]
    assert names == ["weight", "inner.weight", "inner.bias"]
    assert net.scalar == 7
    assert isinstance(net.weight, nn.Parameter)
    assert isinstance(net.inner, nn.Linear)


def test_parameter_defaults_to_requires_grad():
    assert nn.Parameter(np.zeros(3)).requires_grad


def test_nested_module_names_are_qualified():
    model = nn.Sequential(
        nn.Sequential(nn.Linear(2, 3), nn.ReLU()),
        nn.Linear(3, 1),
    )
    assert [n for n, _ in model.named_parameters()] == [
        "0.0.weight", "0.0.bias", "1.weight", "1.bias"
    ]


def test_named_modules_covers_the_whole_tree():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
    assert [n for n, _ in model.named_modules()] == ["", "0", "1"]
    assert len(list(model.children())) == 2


def test_tied_parameters_are_yielded_once():
    """Deduplication by identity: an optimizer must not update a shared
    weight twice per step."""
    shared = nn.Linear(3, 3)

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = shared
            self.b = shared

        def forward(self, x):
            return self.b(self.a(x))

    assert len(list(Tied().parameters())) == 2


def test_buffers_are_state_but_not_parameters():
    class WithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(np.ones(3))
            self.register_buffer("running_mean", tt.Tensor(np.zeros(3)))

    module = WithBuffer()
    assert [n for n, _ in module.named_parameters()] == ["weight"]
    assert [n for n, _ in module.named_buffers()] == ["running_mean"]
    assert sorted(module.state_dict()) == ["running_mean", "weight"]


def test_registering_a_non_parameter_is_rejected():
    module = nn.Module()
    with pytest.raises(TypeError, match="must be a Parameter"):
        module.register_parameter("w", tt.Tensor(np.zeros(3)))


def test_missing_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = nn.Linear(2, 2).nope


def test_num_parameters():
    assert nn.Linear(4, 3).num_parameters() == 4 * 3 + 3


def test_module_zero_grad():
    model = nn.Linear(3, 2)
    model(tt.Tensor(np.ones((5, 3)))).sum().backward()
    assert model.weight.grad is not None
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


# --------------------------------------------------------------------------
# train/eval.
# --------------------------------------------------------------------------
def test_train_eval_propagates_to_every_descendant():
    model = nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 2)))
    assert all(m.training for m in model.modules())
    model.eval()
    assert not any(m.training for m in model.modules())
    model.train()
    assert all(m.training for m in model.modules())


def test_dropout_is_identity_in_eval_and_stochastic_in_train():
    drop = nn.Dropout(0.5)
    x = tt.Tensor(np.ones((200, 20)))

    drop.eval()
    np.testing.assert_array_equal(drop(x).data, x.data)

    drop.train()
    out = drop(x).data
    assert (out == 0).any(), "training-mode dropout should zero some units"
    # Inverted dropout preserves the expected value.
    assert out.mean() == pytest.approx(1.0, abs=0.05)


def test_dropout_gradient_follows_the_mask():
    drop = nn.Dropout(0.5)
    x = tensor64(np.ones((100, 10)))
    out = drop(x)
    out.sum().backward()
    # Gradient is nonzero exactly where the activation survived.
    np.testing.assert_allclose((x.grad != 0), (out.data != 0))


def test_dropout_rejects_invalid_probability():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        nn.Dropout(1.0)


# --------------------------------------------------------------------------
# state_dict / load_state_dict.
# --------------------------------------------------------------------------
def test_state_dict_is_a_snapshot_not_a_view():
    model = nn.Linear(2, 2)
    state = model.state_dict()
    model.weight.data[...] = 99.0
    assert not np.allclose(state["weight"], 99.0)


def test_load_state_dict_round_trip():
    source, destination = nn.Linear(4, 3), nn.Linear(4, 3)
    assert not np.allclose(source.weight.data, destination.weight.data)
    destination.load_state_dict(source.state_dict())
    np.testing.assert_array_equal(destination.weight.data, source.weight.data)
    np.testing.assert_array_equal(destination.bias.data, source.bias.data)


def test_load_state_dict_is_in_place_so_optimizers_keep_working():
    """Loading must not rebind Parameter objects, or an already-constructed
    optimizer would silently update orphaned tensors."""
    model = nn.Linear(3, 2)
    weight_before = model.weight
    model.load_state_dict(nn.Linear(3, 2).state_dict())
    assert model.weight is weight_before


def test_load_state_dict_strict_rejects_mismatch():
    model = nn.Linear(4, 3)
    state = model.state_dict()
    del state["bias"]
    with pytest.raises(KeyError, match="missing"):
        model.load_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == ["bias"] and unexpected == []


def test_load_state_dict_rejects_shape_mismatch():
    model = nn.Linear(4, 3)
    state = model.state_dict()
    state["weight"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="size mismatch"):
        model.load_state_dict(state)


# --------------------------------------------------------------------------
# Layers.
# --------------------------------------------------------------------------
def test_linear_shapes_and_formula():
    layer = nn.Linear(4, 3)
    x = tt.Tensor(np.ones((7, 4)))
    out = layer(x)
    assert out.shape == (7, 3)
    expected = np.ones((7, 4)) @ layer.weight.data.T + layer.bias.data
    np.testing.assert_allclose(out.data, expected)


def test_linear_without_bias_has_no_bias_parameter():
    layer = nn.Linear(4, 3, bias=False)
    assert layer.bias is None
    assert [n for n, _ in layer.named_parameters()] == ["weight"]


def test_linear_supports_extra_leading_dimensions():
    """A (batch, time, features) input should broadcast through the matmul."""
    out = nn.Linear(5, 2)(tt.Tensor(np.ones((3, 6, 5))))
    assert out.shape == (3, 6, 2)


def test_sequential_indexing_and_slicing():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    assert isinstance(model[1], nn.ReLU)
    assert len(model) == 3
    assert len(model[:2]) == 2
    assert isinstance(model[-1], nn.Linear)


def test_sequential_from_ordered_dict_names_layers():
    from collections import OrderedDict

    model = nn.Sequential(OrderedDict([("fc", nn.Linear(2, 2)), ("act", nn.ReLU())]))
    assert [n for n, _ in model.named_parameters()] == ["fc.weight", "fc.bias"]


def test_flatten_and_identity():
    x = tt.Tensor(np.ones((4, 3, 5)))
    assert nn.Flatten()(x).shape == (4, 15)
    assert nn.Identity()(x) is x


def test_module_repr_shows_structure():
    text = repr(nn.Sequential(nn.Linear(2, 3), nn.ReLU()))
    assert "Linear(in_features=2, out_features=3, bias=True)" in text
    assert "ReLU()" in text


def test_base_module_forward_is_abstract():
    with pytest.raises(NotImplementedError):
        nn.Module()(tt.Tensor([1.0]))


# --------------------------------------------------------------------------
# Initialization.
# --------------------------------------------------------------------------
def test_fan_in_and_fan_out_read_off_the_weight_shape():
    assert init.calculate_fan_in_and_fan_out(tt.zeros(3, 5)) == (5, 3)
    # Trailing dims are a receptive field and multiply both fans.
    assert init.calculate_fan_in_and_fan_out(tt.zeros(3, 5, 2, 2)) == (20, 12)


def test_fan_requires_two_dimensions():
    with pytest.raises(ValueError, match="fewer than 2 dimensions"):
        init.calculate_fan_in_and_fan_out(tt.zeros(5))


@pytest.mark.parametrize(
    ("nonlinearity", "expected"),
    [("linear", 1.0), ("sigmoid", 1.0), ("tanh", 5 / 3), ("relu", np.sqrt(2.0))],
)
def test_calculate_gain(nonlinearity, expected):
    assert init.calculate_gain(nonlinearity) == pytest.approx(expected)


def test_calculate_gain_rejects_unknown_nonlinearity():
    with pytest.raises(ValueError, match="unsupported nonlinearity"):
        init.calculate_gain("mystery")


def test_zeros_ones_constant():
    t = tt.empty(3, 4)
    np.testing.assert_array_equal(init.zeros_(t).data, np.zeros((3, 4)))
    np.testing.assert_array_equal(init.ones_(t).data, np.ones((3, 4)))
    np.testing.assert_array_equal(init.constant_(t, 2.5).data, np.full((3, 4), 2.5))


def test_uniform_respects_its_bounds():
    t = init.uniform_(tt.empty(500, 20), -0.25, 0.75)
    assert t.data.min() >= -0.25 and t.data.max() <= 0.75
    assert t.data.mean() == pytest.approx(0.25, abs=0.01)


def test_normal_matches_requested_moments():
    t = init.normal_(tt.empty(600, 40), mean=1.0, std=0.5)
    assert t.data.mean() == pytest.approx(1.0, abs=0.01)
    assert t.data.std() == pytest.approx(0.5, abs=0.01)


def test_xavier_uniform_hits_the_glorot_variance():
    """Var = 2/(fan_in + fan_out); check empirically over a large sample."""
    fan_in, fan_out = 200, 300
    t = init.xavier_uniform_(tt.empty(fan_out, fan_in))
    assert t.data.var() == pytest.approx(2.0 / (fan_in + fan_out), rel=0.05)
    assert abs(t.data).max() <= np.sqrt(6.0 / (fan_in + fan_out)) + 1e-12


def test_xavier_normal_hits_the_glorot_variance():
    fan_in, fan_out = 200, 300
    t = init.xavier_normal_(tt.empty(fan_out, fan_in))
    assert t.data.var() == pytest.approx(2.0 / (fan_in + fan_out), rel=0.05)


def test_kaiming_normal_hits_the_he_variance():
    """With the ReLU gain, Var = 2/fan_in."""
    fan_in, fan_out = 400, 200
    t = init.kaiming_normal_(tt.empty(fan_out, fan_in), nonlinearity="relu")
    assert t.data.var() == pytest.approx(2.0 / fan_in, rel=0.05)


def test_kaiming_uniform_fan_out_mode():
    fan_in, fan_out = 100, 400
    t = init.kaiming_uniform_(tt.empty(fan_out, fan_in), mode="fan_out", nonlinearity="relu")
    assert t.data.var() == pytest.approx(2.0 / fan_out, rel=0.06)


def test_kaiming_rejects_unknown_mode():
    with pytest.raises(ValueError, match="fan_in"):
        init.kaiming_normal_(tt.empty(3, 3), mode="sideways")


def test_default_linear_init_matches_the_pytorch_bound():
    """torch.nn.Linear uses kaiming_uniform_(a=sqrt(5)), i.e. bound 1/sqrt(fan_in)."""
    layer = nn.Linear(64, 32)
    bound = 1.0 / np.sqrt(64)
    assert abs(layer.weight.data).max() <= bound + 1e-12
    assert abs(layer.bias.data).max() <= bound + 1e-12
    # And it is not degenerate: the spread should fill most of the interval.
    assert abs(layer.weight.data).max() > 0.8 * bound


def test_deep_relu_stack_keeps_activation_scale_with_kaiming():
    """The property Kaiming init exists for: activations neither explode nor
    vanish through depth. Xavier, which ignores ReLU's halving, decays."""
    depth, width = 12, 256
    x = tt.Tensor(np.random.default_rng(0).standard_normal((64, width)))

    def propagate(initializer):
        h = x
        for _ in range(depth):
            layer = nn.Linear(width, width, bias=False)
            initializer(layer.weight, nonlinearity="relu") if "kaiming" in initializer.__name__ \
                else initializer(layer.weight)
            h = tt.relu(layer(h))
        return float(h.data.std())

    kaiming_scale = propagate(init.kaiming_normal_)
    xavier_scale = propagate(init.xavier_normal_)
    assert 0.2 < kaiming_scale < 5.0, f"kaiming drifted to {kaiming_scale}"
    assert xavier_scale < kaiming_scale / 5, "xavier should decay faster through ReLU"


def test_manual_seed_makes_initialization_reproducible():
    tt.manual_seed(99)
    first = nn.Linear(8, 6).weight.data.copy()
    tt.manual_seed(99)
    second = nn.Linear(8, 6).weight.data.copy()
    np.testing.assert_array_equal(first, second)
