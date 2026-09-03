"""Optimizer mechanics: update rules, state lifecycle, param groups.

Cross-framework equivalence lives in ``test_torch_parity.py``. This file checks
the things that are TinyTorch's own responsibility: hand-computed first steps,
state creation and checkpointing, group handling and input validation.
"""

from __future__ import annotations

import numpy as np
import pytest

import tinytorch as tt
from tinytorch import nn, optim


def param(values):
    p = nn.Parameter(np.asarray(values, dtype=np.float64))
    return p


def set_grad(p, values):
    p.grad = np.asarray(values, dtype=np.float64)


# --------------------------------------------------------------------------
# SGD, computed by hand.
# --------------------------------------------------------------------------
def test_sgd_step_is_exactly_lr_times_gradient():
    p = param([1.0, 2.0])
    set_grad(p, [0.5, -1.0])
    optim.SGD([p], lr=0.1).step()
    np.testing.assert_allclose(p.data, [1.0 - 0.05, 2.0 + 0.1])


def test_sgd_momentum_first_two_steps_by_hand():
    """Step 1 seeds the buffer with g (not 0.1*g), so it equals plain SGD;
    step 2 is then lr*(mu*g1 + g2)."""
    p = param([0.0])
    opt = optim.SGD([p], lr=0.1, momentum=0.9)

    set_grad(p, [1.0])
    opt.step()
    assert p.data[0] == pytest.approx(-0.1)

    set_grad(p, [1.0])
    opt.step()
    # buffer = 0.9*1 + 1 = 1.9 -> step of 0.19
    assert p.data[0] == pytest.approx(-0.29)


def test_sgd_nesterov_first_step_by_hand():
    """Nesterov adds mu*buf to the gradient: step 1 uses (1 + mu) * g."""
    p = param([0.0])
    opt = optim.SGD([p], lr=0.1, momentum=0.9, nesterov=True)
    set_grad(p, [1.0])
    opt.step()
    assert p.data[0] == pytest.approx(-0.1 * (1.0 + 0.9))


def test_sgd_weight_decay_adds_l2_to_the_gradient():
    p = param([2.0])
    set_grad(p, [0.0])
    optim.SGD([p], lr=0.1, weight_decay=0.5).step()
    # g becomes 0 + 0.5*2 = 1.0
    assert p.data[0] == pytest.approx(2.0 - 0.1)


def test_sgd_dampening_scales_the_incoming_gradient_after_step_one():
    p = param([0.0])
    opt = optim.SGD([p], lr=1.0, momentum=0.5, dampening=0.5)
    set_grad(p, [1.0])
    opt.step()  # buffer = 1.0 (dampening not applied on the first step)
    set_grad(p, [1.0])
    opt.step()  # buffer = 0.5*1.0 + 0.5*1.0 = 1.0
    assert p.data[0] == pytest.approx(-2.0)


def test_sgd_without_momentum_creates_no_state():
    """Plain SGD is stateless; allocating buffers would be wasted memory."""
    p = param([1.0])
    set_grad(p, [1.0])
    opt = optim.SGD([p], lr=0.1)
    opt.step()
    assert opt.state == {}


def test_sgd_rejects_nesterov_without_momentum():
    with pytest.raises(ValueError, match="nesterov"):
        optim.SGD([param([1.0])], lr=0.1, nesterov=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lr": -1.0}, "learning rate"),
        ({"lr": 0.1, "momentum": -0.5}, "momentum"),
        ({"lr": 0.1, "weight_decay": -1.0}, "weight_decay"),
    ],
)
def test_sgd_validates_hyperparameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        optim.SGD([param([1.0])], **kwargs)


# --------------------------------------------------------------------------
# Adam, computed by hand.
# --------------------------------------------------------------------------
def test_adam_first_step_is_almost_exactly_lr():
    """With bias correction, step 1 moves by ~lr regardless of gradient scale --
    that is the whole point of the correction, and the sharpest test of it.

    m1_hat = g, v1_hat = g^2, so the update is lr * g / (|g| + eps) ~ lr.
    """
    for gradient in (1e-4, 1.0, 1e4):
        p = param([0.0])
        set_grad(p, [gradient])
        optim.Adam([p], lr=0.1).step()
        assert p.data[0] == pytest.approx(-0.1, rel=1e-3), f"gradient={gradient}"


def test_adam_without_bias_correction_would_be_wrong():
    """Sanity check on the magnitude: an uncorrected first step would be
    (1-b1)/sqrt(1-b2) ~ 3.2x too small. Confirm we are not doing that."""
    p = param([0.0])
    set_grad(p, [1.0])
    optim.Adam([p], lr=0.1).step()
    uncorrected = 0.1 * (1 - 0.9) / (np.sqrt(1 - 0.999) + 1e-8)
    assert abs(p.data[0]) == pytest.approx(0.1, rel=1e-3)
    assert not np.isclose(abs(p.data[0]), uncorrected, rtol=1e-2)


def test_adam_second_step_by_hand():
    p = param([0.0])
    opt = optim.Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    g1, g2 = 1.0, 2.0

    set_grad(p, [g1])
    opt.step()
    set_grad(p, [g2])
    opt.step()

    m = 0.9 * (0.1 * g1) + 0.1 * g2
    v = 0.999 * (0.001 * g1**2) + 0.001 * g2**2
    expected = -0.1 * (m / (1 - 0.9**2)) / (np.sqrt(v / (1 - 0.999**2)) + 1e-8)
    # step 1 moved by ~-0.1
    first = -0.1 * (0.1 * g1 / (1 - 0.9)) / (np.sqrt(0.001 * g1**2 / (1 - 0.999)) + 1e-8)
    assert p.data[0] == pytest.approx(first + expected, rel=1e-9)


def test_adam_state_is_created_lazily_and_holds_the_step_counter():
    p = param([1.0])
    opt = optim.Adam([p], lr=0.1)
    assert opt.state == {}, "no state before the first step"
    set_grad(p, [1.0])
    opt.step()
    state = opt.state[id(p)]
    assert state["step"] == 1
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    opt.step()
    assert opt.state[id(p)]["step"] == 2


def test_adam_amsgrad_keeps_the_running_maximum():
    p = param([0.0])
    opt = optim.Adam([p], lr=0.1, amsgrad=True)
    set_grad(p, [10.0])
    opt.step()
    peak = opt.state[id(p)]["max_exp_avg_sq"].copy()
    set_grad(p, [1e-6])
    opt.step()
    assert opt.state[id(p)]["max_exp_avg_sq"] >= peak


def test_adamw_decouples_weight_decay():
    """AdamW shrinks the weight by lr*wd directly; Adam routes decay through
    the adaptive denominator, so the two must differ."""
    p_adam, p_adamw = param([1.0]), param([1.0])
    set_grad(p_adam, [0.0])
    set_grad(p_adamw, [0.0])
    optim.Adam([p_adam], lr=0.1, weight_decay=0.1).step()
    optim.AdamW([p_adamw], lr=0.1, weight_decay=0.1).step()
    # With zero gradient, AdamW's move is exactly lr*wd*theta.
    assert p_adamw.data[0] == pytest.approx(1.0 - 0.1 * 0.1 * 1.0, rel=1e-6)
    assert not np.isclose(p_adam.data[0], p_adamw.data[0], rtol=1e-3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lr": -1.0}, "learning rate"),
        ({"betas": (1.5, 0.999)}, "beta1"),
        ({"betas": (0.9, -0.1)}, "beta2"),
        ({"eps": -1e-8}, "epsilon"),
    ],
)
def test_adam_validates_hyperparameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        optim.Adam([param([1.0])], **kwargs)


# --------------------------------------------------------------------------
# Shared optimizer behaviour.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("make", [lambda p: optim.SGD(p, lr=0.1), lambda p: optim.Adam(p, lr=0.1)])
def test_parameters_without_gradient_are_left_alone(make):
    used, unused = param([1.0]), param([5.0])
    set_grad(used, [1.0])
    make([used, unused]).step()
    assert unused.data[0] == 5.0


def test_zero_grad_clears_every_group():
    a, b = param([1.0]), param([1.0])
    set_grad(a, [1.0])
    set_grad(b, [1.0])
    opt = optim.SGD([{"params": [a]}, {"params": [b], "lr": 0.5}], lr=0.1)
    opt.zero_grad()
    assert a.grad is None and b.grad is None


def test_param_groups_carry_their_own_hyperparameters():
    slow, fast = param([0.0]), param([0.0])
    set_grad(slow, [1.0])
    set_grad(fast, [1.0])
    opt = optim.SGD([{"params": [slow], "lr": 0.01}, {"params": [fast]}], lr=1.0)
    opt.step()
    assert slow.data[0] == pytest.approx(-0.01)
    assert fast.data[0] == pytest.approx(-1.0)


def test_group_defaults_are_filled_in():
    p = param([1.0])
    opt = optim.Adam([{"params": [p]}], lr=0.01)
    assert opt.param_groups[0]["betas"] == (0.9, 0.999)
    assert opt.param_groups[0]["eps"] == 1e-8


def test_duplicate_parameter_across_groups_is_rejected():
    p = param([1.0])
    with pytest.raises(ValueError, match="more than one parameter group"):
        optim.SGD([{"params": [p]}, {"params": [p]}], lr=0.1)


def test_empty_parameter_list_is_rejected():
    with pytest.raises(ValueError, match="empty parameter list"):
        optim.SGD([], lr=0.1)


def test_non_trainable_parameter_is_rejected():
    frozen = tt.Tensor(np.ones(3), requires_grad=False)
    with pytest.raises(ValueError, match="requires_grad=False"):
        optim.SGD([frozen], lr=0.1)


def test_non_tensor_is_rejected():
    with pytest.raises(TypeError, match="only optimize Tensors"):
        optim.SGD([np.ones(3)], lr=0.1)


# --------------------------------------------------------------------------
# Optimizer state_dict.
# --------------------------------------------------------------------------
def test_optimizer_state_dict_round_trip_reproduces_the_next_step():
    """Restoring state must reproduce the exact trajectory, momentum and all."""
    def build():
        tt.manual_seed(5)
        model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
        return model, optim.Adam(model.parameters(), lr=0.05)

    x = tt.Tensor(np.random.default_rng(6).standard_normal((8, 3)))
    y = tt.Tensor(np.random.default_rng(7).standard_normal((8, 2)))

    def train(model, opt, steps):
        for _ in range(steps):
            opt.zero_grad()
            nn.MSELoss()(model(x), y).backward()
            opt.step()

    reference, opt_ref = build()
    train(reference, opt_ref, 6)

    partial, opt_partial = build()
    train(partial, opt_partial, 3)
    model_state, opt_state = partial.state_dict(), opt_partial.state_dict()

    resumed, opt_resumed = build()
    resumed.load_state_dict(model_state)
    opt_resumed.load_state_dict(opt_state)
    train(resumed, opt_resumed, 3)

    for (name, a), (_, b) in zip(reference.named_parameters(), resumed.named_parameters(), strict=True):
        np.testing.assert_allclose(b.data, a.data, rtol=1e-12, err_msg=name)


def test_optimizer_state_dict_keys_are_positional_not_identities():
    """``id()`` is meaningless across processes, so state must be re-keyed."""
    p = param([1.0])
    set_grad(p, [1.0])
    opt = optim.Adam([p], lr=0.1)
    opt.step()
    state = opt.state_dict()
    assert list(state["state"]) == ["0"]
    assert state["param_groups"][0]["params"] == [0]
    assert state["param_groups"][0]["lr"] == 0.1


def test_optimizer_state_dict_snapshot_is_independent():
    p = param([1.0])
    set_grad(p, [1.0])
    opt = optim.Adam([p], lr=0.1)
    opt.step()
    snapshot = opt.state_dict()
    before = snapshot["state"]["0"]["exp_avg"].copy()
    opt.step()
    np.testing.assert_array_equal(snapshot["state"]["0"]["exp_avg"], before)


def test_load_state_dict_rejects_a_different_model_layout():
    p = param([1.0])
    opt = optim.SGD([p], lr=0.1)
    other = optim.SGD([param([1.0]), param([2.0])], lr=0.1)
    with pytest.raises(ValueError, match="does not match"):
        opt.load_state_dict(other.state_dict())


def test_optimizer_repr_lists_groups():
    text = repr(optim.Adam([param([1.0])], lr=0.1))
    assert "Adam(" in text and "lr: 0.1" in text


def test_step_runs_without_recording_a_graph():
    """The update itself must not extend the computation graph."""
    p = param([1.0])
    set_grad(p, [1.0])
    optim.Adam([p], lr=0.1).step()
    assert p.grad_fn is None and p.is_leaf


def test_optimizer_updates_dont_leak_into_the_next_backward():
    model = nn.Linear(2, 1)
    opt = optim.SGD(model.parameters(), lr=0.1)
    x = tt.Tensor(np.ones((4, 2)))
    for _ in range(3):
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
    # After zero_grad + one backward, the gradient is the single-pass gradient.
    opt.zero_grad()
    model(x).sum().backward()
    np.testing.assert_allclose(model.bias.grad, [4.0])
