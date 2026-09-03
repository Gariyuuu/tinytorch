"""Completion test: the same network in TinyTorch and PyTorch, side by side.

Builds one MLP in each framework, copies identical weights into both, and
compares -- with numbers printed, not just an assertion:

  1. forward outputs
  2. loss
  3. every parameter gradient
  4. every parameter after one optimizer step (SGD, SGD+momentum, Adam)

Everything runs in float64, where both frameworks execute the same IEEE-754
arithmetic; disagreement beyond a few ulps would mean a genuine difference in
the maths, not rounding.

Run:  python examples/06_pytorch_equivalence.py
"""

from __future__ import annotations

import sys

import numpy as np

import tinytorch as tt
from tinytorch import nn, optim

try:
    import torch
except ImportError:
    sys.exit("This example needs PyTorch:  pip install torch")

SIZES = (6, 16, 12, 4)
BATCH = 32
TOLERANCE = 1e-10


def build() -> tuple[nn.Sequential, torch.nn.Sequential]:
    """Two structurally identical MLPs holding bit-identical weights."""
    rng = np.random.default_rng(1234)
    mine_layers, theirs_layers = [], []
    for i in range(len(SIZES) - 1):
        mine_layers.append(nn.Linear(SIZES[i], SIZES[i + 1]))
        theirs_layers.append(torch.nn.Linear(SIZES[i], SIZES[i + 1], dtype=torch.float64))
        if i < len(SIZES) - 2:
            mine_layers.append(nn.Tanh())
            theirs_layers.append(torch.nn.Tanh())

    mine = nn.Sequential(*mine_layers)
    theirs = torch.nn.Sequential(*theirs_layers)

    linear_mine = [m for m in mine_layers if isinstance(m, nn.Linear)]
    linear_theirs = [m for m in theirs_layers if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lm, lt in zip(linear_mine, linear_theirs, strict=True):
            w = rng.standard_normal(lm.weight.shape) * 0.4
            b = rng.standard_normal(lm.bias.shape) * 0.4
            lm.weight.data[...] = w
            lm.bias.data[...] = b
            lt.weight.copy_(torch.tensor(w, dtype=torch.float64))
            lt.bias.copy_(torch.tensor(b, dtype=torch.float64))
    return mine, theirs


def delta(mine, theirs) -> float:
    """Max absolute difference between a TinyTorch and a PyTorch tensor."""
    a = mine.data if isinstance(mine, tt.Tensor) else np.asarray(mine)
    b = theirs.detach().cpu().numpy()
    return float(np.abs(np.asarray(a, dtype=np.float64) - b).max())


def report(label: str, difference: float) -> bool:
    ok = difference <= TOLERANCE
    print(f"  {label:<34} max|diff| = {difference:.3e}   {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    tt.set_default_dtype(np.float64)
    rng = np.random.default_rng(99)
    x = rng.standard_normal((BATCH, SIZES[0]))
    labels = rng.integers(0, SIZES[-1], size=BATCH)

    print("=" * 72)
    print("TinyTorch <-> PyTorch equivalence")
    print("=" * 72)
    print(f"architecture : {' -> '.join(str(s) for s in SIZES)} (tanh)")
    print(f"batch        : {BATCH}")
    print("dtype        : float64 both sides")
    print(f"tolerance    : {TOLERANCE:.0e} absolute")
    print(f"torch        : {torch.__version__}")

    all_ok = []

    # ---- 1. forward ------------------------------------------------------
    mine, theirs = build()
    x_mine = tt.Tensor(x)
    x_theirs = torch.tensor(x, dtype=torch.float64)
    out_mine, out_theirs = mine(x_mine), theirs(x_theirs)

    print(f"\n[1] forward output  (shape {out_mine.shape})")
    all_ok.append(report("logits", delta(out_mine, out_theirs)))

    # ---- 2. loss ---------------------------------------------------------
    y_mine = tt.Tensor(labels)
    y_theirs = torch.tensor(labels, dtype=torch.long)
    loss_mine = nn.CrossEntropyLoss()(out_mine, y_mine)
    loss_theirs = torch.nn.CrossEntropyLoss()(out_theirs, y_theirs)

    print("\n[2] loss")
    print(f"  {'tinytorch':<34} {loss_mine.item():.15f}")
    print(f"  {'pytorch':<34} {loss_theirs.item():.15f}")
    all_ok.append(report("cross-entropy", abs(loss_mine.item() - loss_theirs.item())))

    # ---- 3. gradients ----------------------------------------------------
    loss_mine.backward()
    loss_theirs.backward()

    print("\n[3] parameter gradients")
    for (name, p_mine), p_theirs in zip(
        mine.named_parameters(), theirs.parameters(), strict=True
    ):
        label = f"grad {name} {tuple(p_mine.shape)}"
        all_ok.append(report(label, delta(p_mine.grad, p_theirs.grad)))

    # ---- 4. one optimizer step, three optimizers -------------------------
    print("\n[4] parameters after one optimizer step")
    optimizers = [
        ("SGD(lr=0.1)", lambda p: optim.SGD(p, lr=0.1), lambda p: torch.optim.SGD(p, lr=0.1)),
        (
            "SGD(lr=0.1, momentum=0.9)",
            lambda p: optim.SGD(p, lr=0.1, momentum=0.9),
            lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),
        ),
        (
            "Adam(lr=0.01)",
            lambda p: optim.Adam(p, lr=0.01),
            lambda p: torch.optim.Adam(p, lr=0.01),
        ),
    ]

    for label, make_mine, make_theirs in optimizers:
        model_mine, model_theirs = build()
        opt_mine = make_mine(list(model_mine.parameters()))
        opt_theirs = make_theirs(list(model_theirs.parameters()))

        opt_mine.zero_grad()
        nn.CrossEntropyLoss()(model_mine(x_mine), y_mine).backward()
        opt_mine.step()

        opt_theirs.zero_grad()
        torch.nn.CrossEntropyLoss()(model_theirs(x_theirs), y_theirs).backward()
        opt_theirs.step()

        worst = max(
            delta(a, b)
            for (_, a), b in zip(
                model_mine.named_parameters(), model_theirs.parameters(), strict=True
            )
        )
        all_ok.append(report(label, worst))

    # ---- 5. ten steps, to catch state drift ------------------------------
    print("\n[5] parameters after 10 Adam steps (catches optimizer-state drift)")
    model_mine, model_theirs = build()
    opt_mine = optim.Adam(model_mine.parameters(), lr=0.01)
    opt_theirs = torch.optim.Adam(model_theirs.parameters(), lr=0.01)
    for step in range(1, 11):
        opt_mine.zero_grad()
        nn.CrossEntropyLoss()(model_mine(x_mine), y_mine).backward()
        opt_mine.step()

        opt_theirs.zero_grad()
        torch.nn.CrossEntropyLoss()(model_theirs(x_theirs), y_theirs).backward()
        opt_theirs.step()

        if step in (1, 2, 5, 10):
            worst = max(
                delta(a, b)
                for (_, a), b in zip(
                    model_mine.named_parameters(), model_theirs.parameters(), strict=True
                )
            )
            all_ok.append(report(f"after step {step:>2}", worst))

    print("\n" + "=" * 72)
    print(f"{sum(all_ok)}/{len(all_ok)} checks passed at {TOLERANCE:.0e}")
    print("=" * 72)
    if not all(all_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
